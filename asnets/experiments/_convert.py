#!/usr/bin/env python3
"""Script for converting experiment files from old architecture format to new
architecture format. Might keep it around in case I need to do another
migration of similar magnitude in the future."""

import ast
import os

from asnets.scripts.run_asnets import opt_str, int_or_float

TRAIN_MOD = 'TRAIN_MODEL_FLAGS'
TEST_MOD = 'TEST_MODEL_FLAGS'


def reprocess_file(file_path):
    out_lines = []
    prev_num_layers = None
    prev_hidden_size = None
    with open(file_path, 'r') as fp:
        for line in fp:
            # strip trailing newline
            if line.endswith('\n'):
                line = line[:-1]

            # parse line
            try:
                parsed = ast.parse(line)
            except SyntaxError:
                parsed = None
            if parsed is None or not parsed.body \
               or not isinstance(parsed.body[0], ast.Assign):
                # skip this line
                out_lines.append(line)
                continue

            if len(parsed.body) > 1:
                raise ValueError(
                    "don't know how to handle multiple statements on one "
                    f"line: '{line}'")

            # if we've gone this far, then body is single Assign statement
            assignment, = parsed.body
            if len(assignment.targets) != 1:
                raise ValueError(
                    f"don't know how to handle multiple targets: {line}")
            dest_name = assignment.targets[0].id
            if dest_name not in {TRAIN_MOD, TEST_MOD}:
                # skip line, again
                out_lines.append(line)
                continue

            # now we'll just assume that the value is a single string (no error
            # checking for numbers, tuples, etc.)
            opts_dict = opt_str(assignment.value.s)
            opts_dict = {k: int_or_float(v) for k, v in opts_dict.items()}
            dropout = opts_dict.setdefault('dropout', None)
            num_layers = opts_dict['num_layers']
            hidden_size = opts_dict['hidden_size']
            if prev_num_layers is not None or prev_hidden_size is not None:
                # Make sure num_layers is consistent between different
                # declarations (e.g TRAIN_MODEL_FLAGS & TEST_MODEL_FLAGS). This
                # might incorrectly flag files where one of those options
                # appears twice with different values (e.g one
                # TRAIN_MODEL_FLAGS is overwritten by another one).
                assert num_layers == prev_num_layers, \
                    (num_layers, prev_num_layers)
                assert hidden_size == prev_hidden_size, \
                    (hidden_size, prev_hidden_size)
            prev_num_layers = num_layers
            prev_hidden_size = hidden_size
            if dest_name == TEST_MOD:
                # we only care about TRAIN_MODEL_FLAGS; TEST_MODEL_FLAGS has
                # no effect any more
                continue
            new_lines = [
                f"NUM_LAYERS = {num_layers}",
                f"HIDDEN_SIZE = {hidden_size}",
            ]
            if dropout is not None:
                new_lines.append(f'DROPOUT = {dropout}')
            out_lines.extend(new_lines)
    final_file = '\n'.join(out_lines)
    if not final_file.endswith('\n'):
        final_file += '\n'
    with open(file_path, 'w') as fp:
        fp.write(final_file)


def main():
    # run reprocess_file on actprop_*.py
    this_dir = os.path.dirname(os.path.abspath(__file__))
    for file_name in os.listdir(this_dir):
        if file_name.startswith('actprop_') and file_name.endswith('.py'):
            print(f"Processing {file_name}")
            file_path = os.path.join(this_dir, file_name)
            reprocess_file(file_path)


if __name__ == '__main__':
    main()
