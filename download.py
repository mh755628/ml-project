#download youtube in wav format using yt-dlp
import yt_dlp
import os
import sys
import time
import subprocess
import shutil
import re

#download youtube in wav format using yt-dlp
def download_youtube(url, output_dir):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_dir + '/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


url = input("Enter the URL of the video you want to download: ")
download_youtube(url, './')