import torch
import torch.nn as nn
import math

class BaselinePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(BaselinePositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
# TODO: Many more positional encodings: Absolute or relative, use Time, use Pitch, or learn the embedding using a custom loss
# TODO: For time/pitch test the effect of bar/octave encoding

class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(TimePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    def forward(self, x, time):
        pe = torch.zeros(x.size(0), self.d_model)
        pe[:, 0::2] = torch.sin(time * self.div_term)
        pe[:, 1::2] = torch.cos(time * self.div_term)
        return x + pe
    
class PitchPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PitchPositionalEncoding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")
        self.d_model = d_model
        self.div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model//2)))

    def forward(self, x, pitch_class, octave):
        pe = torch.zeros(x.size(0), self.d_model)
        pe[:, 0:self.d_model//2:2] = torch.sin(pitch_class * self.div_term)
        pe[:, 1:self.d_model//2:2] = torch.cos(pitch_class * self.div_term)
        
        pe[:, self.d_model//2::2] = torch.sin(octave * self.div_term)
        pe[:, self.d_model//2+1::2] = torch.cos(octave * self.div_term)
        return x + pe
    
class TimeAndPitchPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(TimeAndPitchPositionalEncoding, self).__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be a multiple of 4")
        self.d_model = d_model
        self.time_encoding = TimePositionalEncoding(d_model//2)
        self.pitch_encoding = PitchPositionalEncoding(d_model//2)

    def forward(self, x, time, pitch_class, octave):
        x_time = self.time_encoding(x[:, :self.d_model//2], time)
        x_pitch = self.pitch_encoding(x[:, self.d_model//2:], pitch_class, octave)
        return torch.cat([x_time, x_pitch])



# Vocabulary: Note has an Instrument, and Velocity, and Duration
