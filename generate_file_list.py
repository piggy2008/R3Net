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

def generate_seq(path, save_path, dataset='davis', batch=5):
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
                path_temp = os.path.join(dataset, folder, name)
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
                path_temp = os.path.join('DAFB2', folder, name)
                if j == (i + batch * step - step):
                    image_batch = image_batch + path_temp
                else:
                    image_batch = image_batch + path_temp + ','
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def generate_seq_by_MCL(path, gt_path, save_path, batch=5):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()

        gts = os.listdir(os.path.join(gt_path, folder))
        gts.sort()
        for i in range(0, len(gts)):
            image_batch = ''
            image_name, suffix = os.path.splitext(gts[i])
            for j in range(batch):
                image_index = images.index(image_name + '.jpg')
                image = images[image_index - j]
                print (os.path.join(path, folder, image))
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join(folder, name)
                if j == 0:
                    image_batch = path_temp + image_batch
                else:
                    image_batch = path_temp + ',' + image_batch
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def generate_seq_by_VOS(path, gt_path, save_path, batch=5):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()

        gts = os.listdir(os.path.join(gt_path, folder))
        gts.sort()
        for i in range(0, len(gts)):
            image_batch = ''
            image_name, suffix = os.path.splitext(gts[i])
            for j in range(batch):
                image_index = images.index(image_name + '.png')
                image = images[image_index - j]
                print (os.path.join(path, folder, image))
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join(folder, name)
                if j == 0:
                    image_batch = path_temp + image_batch
                else:
                    image_batch = path_temp + ',' + image_batch
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def generate_seq_by_ViSal(path, gt_path, save_path, batch=5):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()

        gts = os.listdir(os.path.join(gt_path, folder))
        gts.sort()
        for i in range(1, len(gts)):
            image_batch = ''
            image_name, suffix = os.path.splitext(gts[i])
            for j in range(batch):
                image_index = images.index(image_name + '.jpg')
                image = images[image_index - j]
                print (os.path.join(path, folder, image))
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join(folder, name)
                if j == 0:
                    image_batch = path_temp + image_batch
                else:
                    image_batch = path_temp + ',' + image_batch
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def generate_single_by_THUR(path, save_path, batch):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for img in images:
           image_batch = ''
           for i in range(0, batch):
               name, suffix = os.path.splitext(img)
               path_temp = os.path.join('THUR15000', folder, name)
               if i == (batch - 1):
                   image_batch = image_batch + path_temp
               else:
                   image_batch = image_batch + path_temp + ','
           print(image_batch)
           file.writelines(image_batch + '\n')

    file.close()

def generate_seq_by_DUTS_TR(path, save_path):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    images = os.listdir(os.path.join(path, 'DUTS-TR-Image'))
    images.sort()
    for img in images:
       name, suffix = os.path.splitext(img)
       img_temp = os.path.join('DUTS-TR', 'DUTS-TR-Image', img)
       gt_temp = os.path.join('DUTS-TR', 'DUTS-TR-Mask', name + '.png')
       print(img_temp + ' ' + gt_temp)
       file.writelines(img_temp + ' ' + gt_temp + '\n')

    file.close()

def generate_seq_by_DUT_OMRON(path, save_path, batch):
    file = open(save_path, 'w')
    images = os.listdir(path)
    images.sort()
    for img in images:
       image_batch = ''
       for i in range(0, batch):
           name, suffix = os.path.splitext(img)
           path_temp = os.path.join('DUT-OMRON', name)
           if i == (batch - 1):
               image_batch = image_batch + path_temp
           else:
               image_batch = image_batch + path_temp + ','
       print(image_batch)
       file.writelines(image_batch + '\n')

    file.close()

def generate_seq_by_MSRA10K(path, save_path, batch):
    file = open(save_path, 'w')
    images = os.listdir(path)
    images.sort()
    for img in images:
       image_batch = ''
       for i in range(0, batch):
           name, suffix = os.path.splitext(img)
           path_temp = os.path.join('MSRA10K_Imgs_GT', name)
           if i == (batch - 1):
               image_batch = image_batch + path_temp
           else:
               image_batch = image_batch + path_temp + ','
       print(image_batch)
       file.writelines(image_batch + '\n')

    file.close()

if __name__ == '__main__':
    # root = '/home/qub/data/saliency/video_saliency/train_all_gt2_revised/MSRA10K_Imgs_GT'
    # save_path = '/home/qub/data/saliency/video_saliency/train_all_MSRA10K_seq_5f.txt'
    root = '/home/qub/data/saliency/SegTrack-V2/SegTrackV2_test'
    gt_root = '/home/qub/data/saliency/VOS/GT'
    save_path = '/home/qub/data/saliency/SegTrack-V2/SegTrackV2_test_5f.txt'

    # genrate_davis_test_single_imgs(root, save_path)
    # generate_seq_by_DUTS_TR(root, save_path)
    # generate_seq_by_MCL(root, gt_root, save_path)
    generate_seq(root, save_path, dataset='')
    # generate_seq_by_ViSal(root, gt_root, save_path)