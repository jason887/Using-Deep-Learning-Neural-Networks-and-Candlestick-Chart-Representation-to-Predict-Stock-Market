import argparse

def resnet(resnet_model):
    print(len(resnet_model))
    if len(resnet_model) == 2:
        name = resnet_model[0]*resnet_model[1]
        print(name)
        for i in range(0,resnet_model[0]):
            print("a_{}".format(i))
        for i in range(0,resnet_model[1]):
            print("b_{}".format(i))
    if len(resnet_model) == 3:
        name = resnet_model[0]*resnet_model[1]*resnet_model[2]
        print(name)
        for i in range(0,resnet_model[0]):
            print("a_{}".format(i))
        for i in range(0,resnet_model[1]):
            print("b_{}".format(i))
        for i in range(0,resnet_model[2]):
            print("c_{}".format(i))
    if len(resnet_model) == 4:
        name = resnet_model[0]*resnet_model[1]*resnet_model[2]*resnet_model[3]
        print(name)
        for i in range(0,resnet_model[0]):
            print("a_{}".format(i))
        for i in range(0,resnet_model[1]):
            print("b_{}".format(i))
        for i in range(0,resnet_model[2]):
            print("c_{}".format(i))
        for i in range(0,resnet_model[3]):
            print("d_{}".format(i))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',nargs='+', type=int,
                        help='an input directory of dataset', required=True)
    args = parser.parse_args()
    print(args.input)
    layer = args.input
    resnet(layer)

if __name__ == "__main__":
    main()
