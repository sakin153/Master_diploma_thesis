#!/bin/bash
echo "Checking for most recent scene..."
ls -td ~/.cache/world-creator/worlds/*/ 2>/dev/null | head -1
