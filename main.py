import os
import TextProcess
if __name__ == '__main__':
    dir_name = os.path.dirname(__file__)

    read_File_tiny = TextProcess.TextProcessing(dir_name + "\\Data\\hw01_tiny.txt", "tiny.txt")
    read_File_tiny.write_pdf_file(True)


    # read_File_AMemorableFancy = TextProcess.TextProcessing(dir_name + "\\Data\\hw01_AMemorableFancy.txt", "AMemorableFancy.txt")
    # read_File_AMemorableFancy.write_pdf_file(True)

    # read_File_FireFairies = TextProcess.TextProcessing(dir_name + "\\Data\\hw01_FireFairies.txt", "FireFairies.txt")
    # read_File_FireFairies.write_pdf_file(True)





