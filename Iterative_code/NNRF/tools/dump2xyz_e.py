#!/bin/env python
import os, sys
from NNRF.convert import convert_dump_to_xyz

dumpfile=sys.argv[1]
elements=sys.argv[2:]
print(elements)
xyzfilename = dumpfile[:-4] + 'xyz'
Nimages = convert_dump_to_xyz(dump_filename=dumpfile,xyz_filename=xyzfilename, elements=elements)

print(Nimages)
