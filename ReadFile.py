class FileReader(object):
    def __init__(self, filePath):
        self.path = filePath
    def read_txt_file(self):
        with open(self.path) as f:
            contents = f.read()
            print(contents)