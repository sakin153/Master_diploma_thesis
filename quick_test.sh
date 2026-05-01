#!/bin/bash
echo "Running: python main.py 'стол и 3 стула'"
python main.py "стол и 3 стула" 2>&1 | grep -E "\[Stage1\]|\[pipeline\] Objects:|\[pipeline\] ✓ Generated semantic plan|✓ Placed"
