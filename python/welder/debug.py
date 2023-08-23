import sys
import code
import readline
import rlcompleter

"""
This function can start a interactive debug console
example:
    from .debug import debug
    debug()
press CTRL+D to exit the debug console and go on
"""
def debug():
    frame = sys._getframe().f_back
    vars = {**frame.f_locals, **frame.f_globals}
    readline.set_completer(rlcompleter.Completer(vars).complete)
    readline.parse_and_bind("tab: complete")
    banner = f"(DebugConsole) {frame.f_code.co_filename}:{frame.f_lineno}"
    code.InteractiveConsole(vars).interact(banner)
