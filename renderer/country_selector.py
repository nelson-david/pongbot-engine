# engine/renderer/country_selector.py
import sys
from typing import List, Tuple

# Comprehensive sports countries list: (ISO-2 code, Country Name)
COUNTRIES: List[Tuple[str, str]] = [
    ("NG", "Nigeria"),
    ("CN", "China"),
    ("DE", "Germany"),
    ("JP", "Japan"),
    ("FR", "France"),
    ("SE", "Sweden"),
    ("GB", "Great Britain"),
    ("US", "United States"),
    ("KR", "South Korea"),
    ("BR", "Brazil"),
    ("IN", "India"),
    ("TW", "Chinese Taipei"),
    ("SG", "Singapore"),
    ("HK", "Hong Kong"),
    ("EG", "Egypt"),
    ("CA", "Canada"),
    ("AU", "Australia"),
    ("ES", "Spain"),
    ("IT", "Italy"),
    ("PT", "Portugal"),
    ("AT", "Austria"),
    ("RO", "Romania"),
    ("PL", "Poland"),
    ("DK", "Denmark"),
    ("BE", "Belgium"),
    ("NL", "Netherlands"),
    ("AR", "Argentina"),
    ("ZA", "South Africa"),
    ("GH", "Ghana"),
    ("KE", "Kenya"),
    ("CM", "Cameroon"),
]


def _read_key() -> str:
    """Reads a single keypress or escape sequence from terminal in raw mode."""
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch1 = sys.stdin.read(1)
        if ch1 == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return f"\x1b[{ch3}"
            return ch1
        return ch1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def select_country(
    prompt: str = "Select Country",
    default_code: str = "NG",
    visible_rows: int = 5,
) -> str:
    """
    Interactive arrow-key country selector for the CLI.

    Features:
      - Navigate with Up (↑) / Down (↓) arrow keys
      - Quick-jump by typing country letters (e.g. 'N' for Nigeria, 'G' for Great Britain)
      - Press Enter to confirm selection
      - Graceful fallback for non-interactive / piped environments
    """
    # Find initial index matching default_code
    default_code = default_code.upper()
    current_idx = 0
    for i, (code, _) in enumerate(COUNTRIES):
        if code == default_code:
            current_idx = i
            break

    # If stdin or stdout is not a real interactive TTY, fall back to simple text input
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        try:
            choice = input(f"{prompt} (default [{default_code}]): ").strip().upper()
            return choice if choice else default_code
        except (EOFError, KeyboardInterrupt, Exception):
            return default_code

    total = len(COUNTRIES)
    half = visible_rows // 2

    print(f"\n{prompt} [Use ↑/↓ arrows to navigate, Enter to confirm]:")

    def render_menu(first_render: bool = False):
        if not first_render:
            # Move cursor up by visible_rows lines and clear
            sys.stdout.write(f"\x1b[{visible_rows}A")

        start = max(0, min(current_idx - half, total - visible_rows))
        end = min(total, start + visible_rows)
        if end - start < visible_rows:
            start = max(0, end - visible_rows)

        for i in range(start, end):
            code, name = COUNTRIES[i]
            if i == current_idx:
                sys.stdout.write(f"\x1b[2K\r  \x1b[1;35m> [{code}] {name}\x1b[0m\n")
            else:
                sys.stdout.write(f"\x1b[2K\r    [{code}] {name}\n")
        sys.stdout.flush()

    try:
        render_menu(first_render=True)
    except Exception:
        return default_code

    while True:
        try:
            key = _read_key()
        except (KeyboardInterrupt, SystemExit):
            print()
            sys.exit(0)
        except Exception:
            return COUNTRIES[current_idx][0]

        # Arrow Up / 'k'
        if key in ("\x1b[A", "k"):
            current_idx = (current_idx - 1) % total
            render_menu()
        # Arrow Down / 'j'
        elif key in ("\x1b[B", "j"):
            current_idx = (current_idx + 1) % total
            render_menu()
        # Enter key confirms
        elif key in ("\r", "\n", " "):
            code, name = COUNTRIES[current_idx]
            # Clear menu lines and print clean confirmation
            sys.stdout.write(f"\x1b[{visible_rows}A")
            for _ in range(visible_rows):
                sys.stdout.write("\x1b[2K\n")
            sys.stdout.write(f"\x1b[{visible_rows}A")
            sys.stdout.write(f"\r  Selected: \x1b[1;32m[{code}] {name}\x1b[0m\n\n")
            sys.stdout.flush()
            return code
        # Quick letter jump
        elif key.isalpha():
            letter = key.upper()
            # Find next country starting with this letter
            found = False
            for offset in range(1, total + 1):
                idx = (current_idx + offset) % total
                c, n = COUNTRIES[idx]
                if c.startswith(letter) or n.upper().startswith(letter):
                    current_idx = idx
                    found = True
                    break
            if found:
                render_menu()
