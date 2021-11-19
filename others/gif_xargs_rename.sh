#bash

#make GIF:
ffmpeg -f image2 -framerate 24 -i %03d.jpg -loop 0 simpson.gif

convert -delay 0 *.png -loop 0 output.gif

#xargs:
rm |xargs ls --ignore *.useful *

#rename:
find . -name '*.txt' -exec 'sh' '-c' 'mv {} $(sed "s/.txt$/.pts/" <<< {})' ';'
find . -name '*.labels' -exec 'sh' '-c' 'mv {} $(sed "s/.labels$/.seg/" <<< {})' ';'
