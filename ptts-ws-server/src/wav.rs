#![allow(unused)]
use anyhow::{Result, bail};
use rubato::Resampler;
use std::io::prelude::*;

pub trait Sample {
    fn to_i16(&self) -> i16;
}

impl Sample for f32 {
    fn to_i16(&self) -> i16 {
        (self.clamp(-1.0, 1.0) * 32767.0) as i16
    }
}

impl Sample for f64 {
    fn to_i16(&self) -> i16 {
        (self.clamp(-1.0, 1.0) * 32767.0) as i16
    }
}

impl Sample for i16 {
    fn to_i16(&self) -> i16 {
        *self
    }
}
pub fn write_wav_header<W: Write>(
    w: &mut W,
    sample_rate: u32,
    chunk_size: u32,
    data_size: u32,
) -> std::io::Result<()> {
    let n_channels = 1u16;
    let bits_per_sample = 16u16;
    let byte_rate = sample_rate * n_channels as u32 * (bits_per_sample / 8) as u32;
    let block_align = n_channels * (bits_per_sample / 8);

    w.write_all(b"RIFF")?;
    w.write_all(&chunk_size.to_le_bytes())?; // unknown chunk size
    w.write_all(b"WAVE")?;

    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?; // PCM format
    w.write_all(&n_channels.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&bits_per_sample.to_le_bytes())?;

    w.write_all(b"data")?;
    w.write_all(&data_size.to_le_bytes())?; // unknown data size

    Ok(())
}

pub fn write_pcm_in_wav<W: Write, S: Sample>(w: &mut W, samples: &[S]) -> std::io::Result<usize> {
    for sample in samples {
        w.write_all(&sample.to_i16().to_le_bytes())?
    }
    Ok(samples.len() * std::mem::size_of::<i16>())
}

pub fn write_pcm_as_wav<W: Write, S: Sample>(
    w: &mut W,
    samples: &[S],
    sample_rate: u32,
) -> std::io::Result<()> {
    let chunk_size = 12u32 + 24u32 + samples.len() as u32 * 2;
    let data_size = samples.len() as u32 * 2;
    write_wav_header(w, sample_rate, chunk_size, data_size)?;
    write_pcm_in_wav(w, samples)?;
    Ok(())
}

enum Parsed<'a, T> {
    Incomplete,
    Complete(T, &'a [u8]),
}

struct MasterChunk;

impl MasterChunk {
    fn parse(data: &[u8]) -> Result<Parsed<'_, Self>> {
        if data.len() < 12 {
            return Ok(Parsed::Incomplete);
        }
        if &data[0..4] != b"RIFF" {
            bail!("wav-decoder: invalid RIFF header");
        }
        if &data[8..12] != b"WAVE" {
            bail!("wav-decoder: invalid WAVE format");
        }

        Ok(Parsed::Complete(MasterChunk, &data[12..]))
    }
}

#[derive(Debug, Clone)]
struct FmtChunk {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}

impl FmtChunk {
    fn parse(data: &[u8]) -> Result<Parsed<'_, Self>> {
        if data.len() < 24 {
            return Ok(Parsed::Incomplete);
        }
        if &data[0..4] != b"fmt " {
            bail!("wav-decoder: invalid fmt chunk");
        }
        let audio_format = u16::from_le_bytes([data[8], data[9]]);
        let channels = u16::from_le_bytes([data[10], data[11]]);
        let sample_rate = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let bits_per_sample = u16::from_le_bytes([data[22], data[23]]);

        let fmt = Self { audio_format, channels, sample_rate, bits_per_sample };
        Ok(Parsed::Complete(fmt, &data[24..]))
    }
}
