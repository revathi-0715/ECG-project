import wfdb
import os

record_path = os.path.join(
    'C:/Users/HP/Desktop/hii/data/mitdb/x_mitdb',
    '100'
)

record = wfdb.rdrecord(record_path, physical=False)
print("Loaded successfully, signal shape:", record.d_signal.shape)
