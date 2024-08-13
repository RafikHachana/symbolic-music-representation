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
        # pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, data):
        # print(self.pe.shape)
        x = data['x']
        return self.pe[:x.size(1), :].repeat(x.size(0), 1, 1)
    
# TODO: Many more positional encodings: Absolute or relative, use Time, use Pitch, or learn the embedding using a custom loss
# TODO: For time/pitch test the effect of bar/octave encoding

class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=None):
        super(TimePositionalEncoding, self).__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        self.register_buffer('div_term', div_term)


    def forward(self, data):
        x = data['x']
        time = data['time'].unsqueeze(-1)
        pe = torch.zeros(x.size(0), x.size(1), self.d_model).cuda()

        # print("PE", pe.shape)
        # print("TIME", time.shape)
        # print("div term", self.div_term.shape)
        # TODO: Double check that this works properly
        pe[:, :, 0::2] = torch.sin(time * self.div_term.repeat(x.size(0), x.size(1), 1))
        pe[:, :, 1::2] = torch.cos(time * self.div_term.repeat(x.size(0), x.size(1), 1))

        # print(x.shape)
        # print(pe.shape)
        return pe
    
class PitchPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=None):
        super(PitchPositionalEncoding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model//2)))

        self.register_buffer('div_term', div_term)


    def forward(self, data):
        x = data['x']
        # Get the pitch class
        pitch_class = x[:, :, :1]

        octave = data['octave'].unsqueeze(-1)
        pe = torch.zeros(x.size(0), x.size(1), self.d_model).cuda()

        pe[:, :, 0:self.d_model//2:2] = torch.sin(pitch_class * self.div_term.repeat(x.size(0), x.size(1), 1))
        pe[:, :, 1:self.d_model//2:2] = torch.cos(pitch_class * self.div_term.repeat(x.size(0), x.size(1), 1))
        
        pe[:, :, self.d_model//2::2] = torch.sin(octave * self.div_term.repeat(x.size(0), x.size(1), 1))
        pe[:, :, self.d_model//2+1::2] = torch.cos(octave * self.div_term.repeat(x.size(0), x.size(1), 1))
        return pe
    
class TimeAndPitchPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=None):
        super(TimeAndPitchPositionalEncoding, self).__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be a multiple of 4")
        self.d_model = d_model
        self.time_encoding = TimePositionalEncoding(d_model//2)
        self.pitch_encoding = PitchPositionalEncoding(d_model//2)

    def forward(self, data):
        x = data['x']
        x_time = self.time_encoding(data)
        x_pitch = self.pitch_encoding(data)
        # print(x_time.shape)
        # print(x_pitch.shape)
        # print(torch.cat([x_time, x_pitch]).shape)
        return torch.cat([x_time, x_pitch], dim=2)



# Vocabulary: Note has an Instrument, and Velocity, and Duration


positional_encoding_classes = {
    "base": BaselinePositionalEncoding,
    "time": TimePositionalEncoding,
    "pitch": PitchPositionalEncoding,
    "time_pitch": TimeAndPitchPositionalEncoding
}
