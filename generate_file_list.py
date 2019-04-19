import os

def genrate_davis_test_single_imgs(root, save_path):
    file = open(save_path, 'w')
    folders = os.listdir(root)
    for folder in folders:
        imgs = os.listdir(os.path.join(root, folder))
        imgs.sort()
        for img in imgs:
            name, suffix = os.path.splitext(img)
            file.writelines(os.path.join(folder, name) + '\n')
            print (os.path.join(folder, name))

    file.close()

if __name__ == '__main__':
    root = '/home/qub/data/saliency/davis/davis_test2'
    save_path = '/home/qub/data/saliency/davis/davis_test2_single.txt'
    genrate_davis_test_single_imgs(root, save_path)