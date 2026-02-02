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

const WINDOW_SIZE: usize = 4096;
const HOP_SIZE: usize = 512;
const THRESHOLD: f32 = 0.05;

struct AudioMessage {
    note: String,
    octave: i32,
    freq: f32,
    amplitude: f32,
}

fn main() -> Result<(), eframe::Error> {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        if let Err(e) = run_audio_loop(tx) {
            eprintln!("Audio thread error: {}", e);
        }
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([400.0, 300.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Rust Tuner",
        options,
        Box::new(|_cc| Ok(Box::new(TunerApp::new(rx)))),
    )
}

struct TunerApp {
    receiver: Receiver<AudioMessage>,
    current_note: String,
    current_freq: String,
    current_amp: f32,
}

impl TunerApp {
    fn new(receiver: Receiver<AudioMessage>) -> Self {
        Self {
            receiver,
            current_note: "--".to_string(),
            current_freq: "0.0 Hz".to_string(),
            current_amp: 0.0,
        }
    }
}

impl eframe::App for TunerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(msg) = self.receiver.try_iter().last() {
            if msg.amplitude > THRESHOLD {
                self.current_note = format!("{}{}", msg.note, msg.octave);
                self.current_freq = format!("{:.1} Hz", msg.freq);
                self.current_amp = msg.amplitude;
            } else {
                self.current_amp = 0.0;
            }
        } else {
            if self.current_amp > 0.0 {
                self.current_amp *= 0.95;
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(30.0);

                let progress = (self.current_amp * 5.0).clamp(0.0, 1.0);
                ui.add(egui::ProgressBar::new(progress).animate(true));

                ui.add_space(40.0);

                let note_color = if self.current_amp > THRESHOLD {
                    egui::Color32::WHITE
                } else {
                    egui::Color32::from_rgb(80, 80, 80)
                };

                ui.label(
                    egui::RichText::new(&self.current_note)
                        .size(120.0)
                        .strong()
                        .color(note_color),
                );

                ui.add_space(20.0);

                ui.label(
                    egui::RichText::new(&self.current_freq)
                        .size(30.0)
                        .color(egui::Color32::LIGHT_GRAY),
                );
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

            let (max_index, _) = fft_input
                .iter()
                .take(WINDOW_SIZE / 2)
                .enumerate()
                .skip(1)
                .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
                .unwrap_or((0, &Complex::default()));

            let max_freq = max_index as f32 * sample_rate / WINDOW_SIZE as f32;

            if max_freq > 0.0 {
                let note_num = 12.0 * (max_freq / 440.0).log2() + 69.0;
                let note_num_round = note_num.round() as i32;
                let note_idx = note_num_round.rem_euclid(12) as usize;
                let octave = (note_num_round / 12) - 1;

                if sender
                    .send(AudioMessage {
                        note: note_names[note_idx].to_string(),
                        octave,
                        freq: max_freq,
                        amplitude: rms,
                    })
                    .is_err()
                {
                    break;
                }
            }
        } else {
            let _ = sender.send(AudioMessage {
                note: "--".to_string(),
                octave: 0,
                freq: 0.0,
                amplitude: rms,
            });
        }

        if ring_buffer.len() >= WINDOW_SIZE {
            for _ in 0..HOP_SIZE {
                ring_buffer.pop_front();
            }
            thread::sleep(std::time::Duration::from_millis(5));
        }
    }
    Ok(())
}