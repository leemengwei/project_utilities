/home/zzp/video_labeler下：

准备阶段：
rm output/labels.txt删除以前的视频标注程序输出

第一步：
./start.sh 12    其中12是文件夹序号，打开视频标定程序。操作指南:
Manipulation

    Select a label from the label GUI windows.
    Press-Hold the left bottom of your mouse until the label name appear then drug a rectangle.
    Right Click your mouse in the box rectangle will remove that box.
    Press Space will stop the video play, so you can clearly view your label.
    Press Esc will exit the labeler.
注意：多目标标定可以建议分多次执行，label会叠加在后面，此时请不要执行rm output/labels.txt

第二部：
python video_labeler_to_labelimg.py 12    转换之前的文件到labelimage格式。12是文件夹序号

第三部：
labelImg  可以打开了
