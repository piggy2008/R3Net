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

def generate_seq(path, save_path, batch):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for i in range(0, len(images) - batch + 1):
            image_batch = ''
            for j in range(batch):

                image = images[i + j]
                print (os.path.join(path, folder, image))
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join('DAFB3', folder, name)
                if j == (batch - 1):
                    image_batch = image_batch + path_temp
                else:
                    image_batch = image_batch + path_temp + ','
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def generate_seq_with_step(path, save_path, batch, step):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for i in range(0, len(images) - batch * step + 1):
            image_batch = ''
            for j in range(i, i + batch * step, step):

                image = images[j]
                print (os.path.join(path, folder, image))
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join(folder, name)
                if j == (i + batch * step - step):
                    image_batch = image_batch + path_temp
                else:
                    image_batch = image_batch + path_temp + ','
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

if __name__ == '__main__':
    root = '/home/qub/data/saliency/video_saliency/train_all/DAFB3'
    save_path = '/home/qub/data/saliency/video_saliency/train_all_DAFB3_seq_5f.txt'
    # genrate_davis_test_single_imgs(root, save_path)
    generate_seq(root, save_path, 5)