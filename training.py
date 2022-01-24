from SOMToolBox_Parse import SOMToolBox_Parse
from minisom import MiniSom

idata = SOMToolBox_Parse("./datasets/chainlink/chainlink.vec").read_weight_file()

som = MiniSom(10, 10, 3, sigma=0.3, learning_rate=0.5)
som.train(idata['arr'], 1000)
weights = som._weights.reshape(-1,3)

weights.tofile("./datasets/weights_chainlink.npy", format="%s")
