import os
import TextProcess
if __name__ == '__main__':
    dir_name = os.path.dirname(__file__)
    read_File_AMemorableFancy = TextProcess.TextProcessing(dir_name + "\\Data\\hw01_AMemorableFancy.txt", "AMemorableFancy.txt")
    read_File_AMemorableFancy.show_results()
    # read_File_FireFairies = ReadFile.FileReader(dir_name + "\\Data\\hw01_FireFairies.txt")
    # read_File_FireFairies.read_txt_file()
    #
    # read_File_tiny = ReadFile.FileReader(dir_name + "\\Data\\hw01_tiny.txt")
    # read_File_AMemorableFancy.read_txt_file()



