import textract


def read_All_CV(filename):
    text = textract.process(filename)
    return text.decode('utf-8')


default_path = "/home/seemsred/Desktop/Hackathon DS/HackDS/cvproj/cv/media/"
filename = default_path + "Абдильдин Аян Абубакирович.pdf"
file = read_All_CV(filename)
print(file)

