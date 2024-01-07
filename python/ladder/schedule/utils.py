import os
import sys
import inspect
dump_schedule = False
tmp_path = './.tmp/scheduled'
count = 0

def enable_schedule_dump():
    global dump_schedule
    dump_schedule = True

def init_count():
    global count
    count = 0
    return count

def write_code(code, path, fname):
    global dump_schedule
    global count
    if not dump_schedule:
        return
    # if path not exist, create it
    fname = str(count) + "." + fname
    count += 1
    if not os.path.exists(path):
        os.makedirs(path)
    # join path and fname
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)


def write_sch_with_path(sch, path, fname):
    global dump_schedule
    if not dump_schedule:
        return
    py_fname = fname + ".py"
    write_code(sch.mod["main"].script(), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(sch.mod.astext(), path, cu_fname)

def write_sch(*args):
    if len(args) == 3:
        return write_sch_with_path(args[0], args[1], args[2])
    elif len(args) == 2:
        global tmp_path
        # get file name and remove the suffix
        fname = os.path.abspath(inspect.getsourcefile(sys._getframe(1)))
        _, fname = os.path.split(fname)
        fname = fname.split(".")[0]
        # create log path
        fname = os.path.join(tmp_path, fname)
        return write_sch_with_path(args[0], fname, args[1])
    else:
        raise ValueError("Invalid arguments for write_sch")
