#!/bin/bash
# Script to run generation and view the scene

echo "=========================================="
echo "Running Quick Test"
echo "=========================================="
python quick_test.py

if [ $? -ne 0 ]; then
    echo "❌ Quick test failed. Exiting."
    exit 1
fi

echo ""
echo "=========================================="
echo "Generating Scene: стол и стул"
echo "=========================================="
python main.py "стол и стул"

if [ $? -ne 0 ]; then
    echo "❌ Scene generation failed."
    exit 1
fi

echo ""
echo "=========================================="
echo "Viewing Scene"
echo "=========================================="
python view_scene.py

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
