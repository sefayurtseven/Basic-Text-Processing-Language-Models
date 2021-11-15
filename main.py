import os
import ReadFile
if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    read_File_AMemorableFancy = ReadFile.FileReader(dirname + "\\Data\\hw01_AMemorableFancy.txt")
    read_File_AMemorableFancy.read_txt_file()

    read_File_FireFairies = ReadFile.FileReader(dirname + "\\Data\\hw01_FireFairies.txt")
    read_File_FireFairies.read_txt_file()

    read_File_tiny = ReadFile.FileReader(dirname + "\\Data\\hw01_tiny.txt")
    read_File_AMemorableFancy.read_txt_file()



