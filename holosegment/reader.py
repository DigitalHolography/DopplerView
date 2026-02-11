"""
Reader module for .holo files with header and footer
"""

import struct
import numpy as np
from pathlib import Path


class HoloReader:
    """
    Reader for .holo files containing doppler hologram data.
    
    The .holo format consists of:
    - Header: metadata about the file
    - Data: raw frame data
    - Footer: additional metadata
    """
    
    def __init__(self, filepath):
        """
        Initialize HoloReader
        
        Args:
            filepath: Path to .holo file
        """
        self.filepath = Path(filepath)
        self.header = {}
        self.footer = {}
        
    def read_header(self, file_handle):
        """
        Read header from .holo file
        
        Expected header format:
        - Magic number (4 bytes): 'HOLO'
        - Version (4 bytes): uint32
        - Width (4 bytes): uint32
        - Height (4 bytes): uint32
        - Num frames (4 bytes): uint32
        - Data type (4 bytes): uint32 (0=uint8, 1=uint16, 2=float32, 3=float64)
        - Header size (4 bytes): uint32
        
        Args:
            file_handle: Open file handle
        """
        # Read magic number
        magic = file_handle.read(4).decode('ascii')
        if magic != 'HOLO':
            raise ValueError(f"Invalid .holo file: magic number '{magic}' != 'HOLO'")
        
        # Read header fields
        version, width, height, num_frames, data_type, header_size = struct.unpack('<6I', file_handle.read(24))
        
        self.header = {
            'magic': magic,
            'version': version,
            'width': width,
            'height': height,
            'num_frames': num_frames,
            'data_type': data_type,
            'header_size': header_size,
        }
        
        # Skip remaining header bytes if any
        remaining = header_size - 28  # 4 bytes for magic + 24 bytes for fields
        if remaining > 0:
            file_handle.read(remaining)
        
        return self.header
    
    def read_footer(self, file_handle):
        """
        Read footer from .holo file
        
        Expected footer format:
        - Footer size (4 bytes): uint32
        - Timestamp (8 bytes): float64
        - Checksum (4 bytes): uint32
        
        Args:
            file_handle: Open file handle
        """
        # Read footer size first
        footer_size_bytes = file_handle.read(4)
        if len(footer_size_bytes) < 4:
            # No footer or incomplete
            return {}
        
        footer_size = struct.unpack('<I', footer_size_bytes)[0]
        
        # Read timestamp and checksum
        if footer_size >= 16:
            timestamp, checksum = struct.unpack('<dI', file_handle.read(12))
            self.footer = {
                'footer_size': footer_size,
                'timestamp': timestamp,
                'checksum': checksum,
            }
        
        return self.footer
    
    def get_dtype(self, data_type):
        """Convert data type code to numpy dtype"""
        dtype_map = {
            0: np.uint8,
            1: np.uint16,
            2: np.float32,
            3: np.float64,
        }
        return dtype_map.get(data_type, np.float32)
    
    def read_frames(self):
        """
        Read all frames from .holo file
        
        Returns:
            numpy array of shape (num_frames, height, width) containing frame data
        """
        with open(self.filepath, 'rb') as f:
            # Read header
            header = self.read_header(f)
            
            # Get frame dimensions
            width = header['width']
            height = header['height']
            num_frames = header['num_frames']
            dtype = self.get_dtype(header['data_type'])
            
            # Calculate expected data size
            frame_size = width * height
            total_size = frame_size * num_frames
            
            # Read frame data
            data = np.fromfile(f, dtype=dtype, count=total_size)
            
            # Reshape to frames
            frames = data.reshape(num_frames, height, width)
            
            # Try to read footer
            try:
                self.read_footer(f)
            except Exception:
                # Footer is optional
                pass
        
        return frames
    
    def get_metadata(self):
        """Get header and footer metadata"""
        return {
            'header': self.header,
            'footer': self.footer,
        }
