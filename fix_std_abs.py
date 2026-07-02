import sys

with open('/usr/include/c++/12/bits/std_abs.h', 'r') as f:
    content = f.read()

# We want to remove the 'namespace std {' we appended at the end
# and instead put our code inside the include guard.
# Let's completely restore it to pristine condition first.
content = content.split('\nnamespace std {\n#if defined(__CUDACC__)')[0]

# Now let's inject our fix right before the #endif // _GLIBCXX_BITS_STD_ABS_H
fix = """
#if defined(__CUDACC__)
namespace std _GLIBCXX_VISIBILITY(default) {
_GLIBCXX_BEGIN_NAMESPACE_VERSION
  inline _GLIBCXX_CONSTEXPR double abs(double __x) { return __builtin_fabs(__x); }
  inline _GLIBCXX_CONSTEXPR float abs(float __x) { return __builtin_fabsf(__x); }
  inline _GLIBCXX_CONSTEXPR long double abs(long double __x) { return __builtin_fabsl(__x); }
_GLIBCXX_END_NAMESPACE_VERSION
}
#endif
"""

# Let's make sure we only insert it once
if 'namespace std _GLIBCXX_VISIBILITY(default) {' not in content.split('#if defined(__CUDACC__)')[-1]:
    # Replace the last #endif with our fix + #endif
    parts = content.rsplit('#endif // _GLIBCXX_BITS_STD_ABS_H', 1)
    if len(parts) == 2:
        content = parts[0] + fix + '\n#endif // _GLIBCXX_BITS_STD_ABS_H' + parts[1]

with open('/usr/include/c++/12/bits/std_abs.h', 'w') as f:
    f.write(content)

print("std_abs.h fixed!")
