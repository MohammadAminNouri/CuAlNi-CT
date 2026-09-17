from __future__ import annotations
import argparse
from pathlib import Path
from .report import write_all_reports
from .symbolic import pretty_symbolic_reference


def main():
    p=argparse.ArgumentParser(description="Cu-Al-Ni crystallography research toolkit")
    sub=p.add_subparsers(dest='cmd',required=True)
    r=sub.add_parser('report',help='generate exact/reference markdown reports')
    r.add_argument('--out',default='reports')
    sub.add_parser('symbolic',help='print symbolic pulled-back metrics')
    a=p.parse_args()
    if a.cmd=='report':
        for q in write_all_reports(Path(a.out)): print(q)
    else:
        print(pretty_symbolic_reference())
if __name__=='__main__': main()
