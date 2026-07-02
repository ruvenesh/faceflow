import sys
import struct

def clear_execstack(filepath):
    with open(filepath, 'rb+') as f:
        # Read ELF header
        magic = f.read(4)
        if magic != b'\x7fELF':
            print("Not an ELF file")
            return
        
        file_class = f.read(1)[0] # 1 for 32-bit, 2 for 64-bit
        f.seek(28 if file_class == 1 else 32)
        e_phoff = struct.unpack('<I' if file_class == 1 else '<Q', f.read(4 if file_class == 1 else 8))[0]
        
        f.seek(42 if file_class == 1 else 54)
        e_phentsize = struct.unpack('<H', f.read(2))[0]
        e_phnum = struct.unpack('<H', f.read(2))[0]
        
        for i in range(e_phnum):
            f.seek(e_phoff + i * e_phentsize)
            p_type = struct.unpack('<I', f.read(4))[0]
            if p_type == 0x6474e551: # PT_GNU_STACK
                # It's PT_GNU_STACK
                f.seek(e_phoff + i * e_phentsize + (24 if file_class == 1 else 4))
                p_flags = struct.unpack('<I', f.read(4))[0]
                # clear PF_X (1)
                new_flags = p_flags & ~1
                f.seek(-4, 1)
                f.write(struct.pack('<I', new_flags))
                print(f"Cleared execstack flag in {filepath}")
                return
        print("PT_GNU_STACK not found")

import glob
for f in glob.glob("vface_venv/lib/python3.10/site-packages/bezier/libbezier-*.so.*"):
    clear_execstack(f)
