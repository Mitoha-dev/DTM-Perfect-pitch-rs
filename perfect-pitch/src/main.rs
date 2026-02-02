use anyhow;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use num_complex::Complex;
use num_traits::Zero;
use rustfft::FftPlanner;
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

const WINDOW_SIZE: usize = 8192;
const HOP_SIZE: usize = 1024;
const THRESHOLD: f32 = 0.02;

struct AudioMessage {
    note_name: String,
    octave: i32,
    freq: f32,
    deviation_cents: f32,
    amplitude: f32,
}

struct TunerApp {
    receiver: Receiver<AudioMessage>,
    display_note: String,
    display_freq: String,
    display_cents: f32,
    display_amp: f32,
    smoothed_cents: f32,
}

fn main() -> Result<(), eframe::Error> {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        if let Err(e) = run_audio_loop(tx) {
            eprintln!("Audio Error: {}", e);
        }
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([500.0, 400.0]),
        ..Default::default()
    };

    eframe::run_native(
        "High-Precision Rust Tuner",
        options,
        Box::new(|_cc| Ok(Box::new(TunerApp::new(rx)))),
    )
}

impl TunerApp {
    fn new(receiver: Receiver<AudioMessage>) -> Self {
        Self {
            receiver,
            display_note: "--".to_string(),
            display_freq: "0.0 Hz".to_string(),
            display_cents: 0.0,
            display_amp: 0.0,
            smoothed_cents: 0.0,
        }
    }
}

impl eframe::App for TunerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(msg) = self.receiver.try_iter().last() {
            if msg.amplitude > THRESHOLD {
                self.display_note = format!("{}{}", msg.note_name, msg.octave);
                self.display_freq = format!("{:.1} Hz", msg.freq);
                self.display_cents = msg.deviation_cents;
                self.display_amp = msg.amplitude;
            } else {
                self.display_amp *= 0.9;
            }
        } else {
            self.display_amp *= 0.95;
        }

        self.smoothed_cents += (self.display_cents - self.smoothed_cents) * 0.2;

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(20.0);

                let canvas_height = 40.0;
                let (response, painter) =
                    ui.allocate_painter(egui::Vec2::new(300.0, canvas_height), egui::Sense::hover());

                let rect = response.rect;
                painter.rect_filled(rect, 5.0, egui::Color32::from_gray(50));

                let center_x = rect.center().x;
                painter.line_segment(
                    [
                        egui::Pos2::new(center_x, rect.top()),
                        egui::Pos2::new(center_x, rect.bottom()),
                    ],
                    egui::Stroke::new(2.0, egui::Color32::WHITE),
                );

                let clamped_cents = self.smoothed_cents.clamp(-50.0, 50.0);
                let offset = (clamped_cents / 50.0) * (rect.width() / 2.0);
                let needle_x = center_x + offset;

                let needle_color = if self.smoothed_cents.abs() < 5.0 {
                    egui::Color32::GREEN
                } else {
                    egui::Color32::RED
                };

                painter.circle_filled(
                    egui::Pos2::new(needle_x, rect.center().y),
                    10.0,
                    needle_color,
                );

                ui.add_space(10.0);

                ui.label(
                    egui::RichText::new(format!("{:.1} cents", self.smoothed_cents))
                        .size(20.0)
                        .color(needle_color),
                );

                ui.add_space(30.0);

                let text_color = if self.display_amp > THRESHOLD {
                    if self.smoothed_cents.abs() < 5.0 {
                        egui::Color32::GREEN
                    } else {
                        egui::Color32::WHITE
                    }
                } else {
                    egui::Color32::DARK_GRAY
                };

                ui.label(
                    egui::RichText::new(&self.display_note)
                        .size(140.0)
                        .strong()
                        .color(text_color),
                );

                ui.add_space(20.0);

                ui.label(
                    egui::RichText::new(&self.display_freq)
                        .size(24.0)
                        .color(egui::Color32::GRAY),
                );

                ui.add_space(20.0);

                let progress = (self.display_amp * 5.0).clamp(0.0, 1.0);
                ui.add(egui::ProgressBar::new(progress).animate(true));
            });
        });

        ctx.request_repaint();
    }
}

fn run_audio_loop(sender: Sender<AudioMessage>) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("マイクなし");
    let config: cpal::StreamConfig = device.default_input_config()?.into();
    let sample_rate = config.sample_rate.0 as f32;

    let (audio_tx, audio_rx) = mpsc::channel::<f32>();

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            for &sample in data {
                let _ = audio_tx.send(sample);
            }
        },
        |err| eprintln!("Stream error: {}", err),
        None,
    )?;
    stream.play()?;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(WINDOW_SIZE);
    let mut ring_buffer: VecDeque<f32> = VecDeque::with_capacity(WINDOW_SIZE);
    let mut fft_input: Vec<Complex<f32>> = vec![Complex::zero(); WINDOW_SIZE];

    let window: Vec<f32> = (0..WINDOW_SIZE)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (WINDOW_SIZE as f32 - 1.0)).cos()))
        .collect();

    let note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

    loop {
        while let Ok(sample) = audio_rx.try_recv() {
            ring_buffer.push_back(sample);
            if ring_buffer.len() > WINDOW_SIZE {
                ring_buffer.pop_front();
            }
        }
        if ring_buffer.len() < WINDOW_SIZE {
            if let Ok(sample) = audio_rx.recv() {
                ring_buffer.push_back(sample);
            }
            continue;
        }

        for (i, (&sample, &win)) in ring_buffer.iter().zip(&window).enumerate() {
            fft_input[i] = Complex { re: sample * win, im: 0.0 };
        }

        let rms: f32 =
            (fft_input.iter().map(|c| c.re * c.re).sum::<f32>() / WINDOW_SIZE as f32).sqrt();

        if rms > THRESHOLD {
            fft.process(&mut fft_input);

            let spectrum: Vec<f32> =
                fft_input.iter().take(WINDOW_SIZE / 2).map(|c| c.norm()).collect();

            let (max_idx, _) = spectrum
                .iter()
                .enumerate()
                .skip(1)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap_or((0, &0.0));

            let interpolated_freq = if max_idx > 0 && max_idx < spectrum.len() - 1 {
                let y_l = spectrum[max_idx - 1];
                let y_c = spectrum[max_idx];
                let y_r = spectrum[max_idx + 1];

                let delta = 0.5 * (y_l - y_r) / (y_l - 2.0 * y_c + y_r);
                (max_idx as f32 + delta) * sample_rate / WINDOW_SIZE as f32
            } else {
                max_idx as f32 * sample_rate / WINDOW_SIZE as f32
            };

            if interpolated_freq > 20.0 {
                let note_number = 12.0 * (interpolated_freq / 440.0).log2() + 69.0;
                let nearest_note_num = note_number.round() as i32;

                let deviation = (note_number - nearest_note_num as f32) * 100.0;

                let note_idx = nearest_note_num.rem_euclid(12) as usize;
                let octave = (nearest_note_num / 12) - 1;

                let _ = sender.send(AudioMessage {
                    note_name: note_names[note_idx].to_string(),
                    octave,
                    freq: interpolated_freq,
                    deviation_cents: deviation,
                    amplitude: rms,
                });
            }
        } else {
            let _ = sender.send(AudioMessage {
                note_name: "--".to_string(),
                octave: 0,
                freq: 0.0,
                deviation_cents: 0.0,
                amplitude: 0.0,
            });
        }

        if ring_buffer.len() >= WINDOW_SIZE {
            for _ in 0..HOP_SIZE {
                ring_buffer.pop_front();
            }
            thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}