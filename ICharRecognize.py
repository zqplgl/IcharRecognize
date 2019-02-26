import models.crnn as crnn
import torch
import utils
import dataset
from torch.autograd import Variable
from PIL import Image

class CharRecognize:
    def __init__(self,weightfile,gpu_id=0):
        self.__net = crnn.CRNN(32,1,37,256)
        if torch.cuda.is_available():
            self.__net.cuda(device=gpu_id)
            self.__gpu_id = gpu_id

        self.__net.load_state_dict(torch.load(weightfile))
        alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.__converter = utils.strLabelConverter(alphabet)
        self.__transformer = dataset.resizeNormalize((100,32))

    def recognize(self,im):
        im_gray = im.convert("L")
        img = self.__transformer(im_gray)
        if torch.cuda.is_available():
            img = img.cuda(device=self.__gpu_id)
        img = img.view(1, *img.size())
        img = Variable(img)
        self.__net.eval()
        preds = self.__net(img)
        _,preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = self.__converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.__converter.decode(preds.data, preds_size.data, raw=False)
        # print('%-20s => %-20s' % (raw_pred, sim_pred))

        return sim_pred

def run():
    model_path = './data/crnn.pth'

    recognizer = CharRecognize(model_path)
    img_path = './data/test3.jpg'

    im = Image.open(img_path)
    result = recognizer.recognize(im)
    # im.show()
    print result


if __name__=="__main__":
    run()
