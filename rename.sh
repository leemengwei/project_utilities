#!/bin/bash

find . -name '*.txt' -exec 'sh' '-c' 'mv {} $(sed "s/.txt$/.pts/" <<< {})' ';'

find . -name '*.labels' -exec 'sh' '-c' 'mv {} $(sed "s/.labels$/.seg/" <<< {})' ';'
