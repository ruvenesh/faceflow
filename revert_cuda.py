import re
import os

files_to_patch = [
    '/usr/local/cuda-12.8/include/crt/math_functions.h',
    '/usr/local/cuda-12.8/include/crt/math_functions.hpp'
]

for path in files_to_patch:
    if not os.path.exists(path): continue
    with open(path, 'r') as f:
        content = f.read()

    # Remove noexcept(true) everywhere it was added to our funcs
    content = content.replace(' noexcept(true)', '')

    # Restore abs/labs/llabs
    pattern_abs = r"/\*\s*(extern\s+__DEVICE_FUNCTIONS_DECL__\s+__device_builtin__\s+__cudart_builtin__\s+(?:int|long\s+int|long\s+long\s+int)\s+__cdecl\s+(?:abs|labs|llabs)\s*\([^)]+\)\s+__THROW\s*;)\s*\*/"
    content = re.sub(pattern_abs, r"\1", content)

    with open(path, 'w') as f:
        f.write(content)

print("Reverted math_functions.h and .hpp to original!")
