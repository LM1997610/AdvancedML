

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_item_by_position(folder_path, position):

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise ValueError("Invalid folder path")

    files = sorted(os.listdir(folder_path))
    if not (0 <= position < len(files)): raise ValueError("Invalid position")

    selected_item_path = os.path.join(folder_path, files[position])
    return selected_item_path

def do_plot(pos, ground_true_path, masks_path, 
            baseline_img_path, our_generated_img_path, labels = False, save= False):

  gt_selected_item_path = get_item_by_position(ground_true_path, pos)
  mask_tyhis_item = get_item_by_position(masks_path, pos)     ## <--- be sure small or large masks
  bs_selected_item_path = get_item_by_position(baseline_img_path, pos)
  our_selected_item_path = get_item_by_position(our_generated_img_path, pos)  ## <---- change hereee


  gt_img = mpimg.imread(gt_selected_item_path)
  mask_img = mpimg.imread(mask_tyhis_item)
  bs_img = mpimg.imread(bs_selected_item_path)
  our_img = mpimg.imread(our_selected_item_path)

  fig, ax = plt.subplots(1, 4, figsize=(8, 8))

  ax[0].imshow(gt_img)
  ax[0].axis('off')

  ax[1].imshow(gt_img, cmap='gray', interpolation='none')
  ax[1].imshow(mask_img, cmap='jet', alpha=0.75*(mask_img<1),interpolation='none')
  ax[1].axis('off')

  ax[2].imshow(bs_img)
  ax[2].axis('off')

  ax[3].imshow(our_img)
  
  ax[3].axis('off')

  if labels == True: 
    ax[0].set_title('\n Ground True', fontsize=14)
    ax[1].set_title('\n Masked', fontsize=12)
    ax[2].set_title('\n Baseline', fontsize=12)
    ax[3].set_title('\n Proposed', fontsize=12)

  plt.subplots_adjust(wspace=0.05)
  if save == True:
     if not os.path.exists("comparison"):
        os.makedir("comparison")
     plt.savefig("comparison/comparison_"+str(pos)+".png")
  plt.show()



def do_plot_batch(images_to_show, ground_true_path, masks_path, baseline_img_path, our_generated_img_path):

  columns = 4
  tot = len(images_to_show)*columns
  rows = tot // columns

  fig, ax = plt.subplots(len(images_to_show), columns, figsize=(8, 8))

  plt.subplots_adjust(wspace=0.03, hspace=0.05)
  
  for img_idx in range(len(images_to_show)):
      
      gt_selected_item_path = get_item_by_position(ground_true_path, images_to_show[img_idx])
      mask_tyhis_item = get_item_by_position(masks_path, images_to_show[img_idx])
      bs_selected_item_path = get_item_by_position(baseline_img_path, images_to_show[img_idx])
      our_selected_item_path = get_item_by_position(our_generated_img_path, images_to_show[img_idx])

      gt_img = mpimg.imread(gt_selected_item_path)
      mask_img = mpimg.imread(mask_tyhis_item)
      bs_img = mpimg.imread(bs_selected_item_path)
      our_img = mpimg.imread(our_selected_item_path)

      for i in range(columns):
        if i == 0:
          ax[img_idx][i].imshow(gt_img, aspect = "auto")
          ax[img_idx][i].axis('off')

        if i == 1:
          ax[img_idx][i].imshow(gt_img, cmap='gray', interpolation='none', aspect = "auto")
          ax[img_idx][i].imshow(mask_img, cmap='jet', alpha=0.75*(mask_img<1),interpolation='none', aspect = "auto")
          ax[img_idx][i].axis('off')
        if i == 2:
          ax[img_idx][i].imshow(bs_img,aspect = "auto")
          ax[img_idx][i].axis('off')

        if i == 3:
          ax[img_idx][i].imshow(our_img, aspect = "auto")
          ax[img_idx][i].axis('off')

  ax[0][0].set_title('\n Ground True', fontsize=14)
  ax[0][1].set_title('\n Masked', fontsize=12)
  ax[0][2].set_title('\n Baseline', fontsize=12)
  ax[0][3].set_title('\n Proposed', fontsize=12)

  

  if not os.path.exists("comparison"):
    os.makedirs("comparison")

  name = '_'.join(str(x) for x in images_to_show)
  plt.savefig(f"comparison/comparison_batch_{name}.png")   
  plt.show()


# do_plot(pos = 122, 
#         ground_true_path ='/content/my_MAT/train_val_test_split/test_sets/test',
#         masks_path = '/content/my_MAT/train_val_test_split/test_sets/masks_small_256', ## <--- be sure small or large masks
#         baseline_img_path = '/content/my_MAT/images_baseline', 
#         our_generated_img_path = '/content/my_MAT/images_run_16kimg', labels =False)