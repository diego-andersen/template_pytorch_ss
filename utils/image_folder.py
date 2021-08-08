import os

IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tiff', '.webp'
]


def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)


def get_img_filepaths(root_dir, recursively=False, read_cache=False, write_cache=False):
    """
    Return a list of filepaths to all images in a directory. Optionally write the
    list to disk for faster retrieval later, or read an existing list from disk.

    Parameters:
        root_dir {str}: Root directory to act on.
        recursively {bool}: Include all sub-directories as well.
        read_cache {bool}: Read filepaths from previously saved text file.
        write_cache {bool}: Write filepaths to text file for faster loading later.

    Returns:
        {list}: List of {str} filepaths to each image in the dataset.
    """

    img_filepaths = []

    if read_cache:
        possible_filelist = os.path.join(root_dir, 'files.list')
        if os.path.isfile(possible_filelist):
            with open(possible_filelist, 'r') as f:
                img_filepaths = f.read().splitlines()
            return img_filepaths

    assert os.path.isdir(root_dir) or os.path.islink(root_dir), '{} is not a valid directory'.format(root_dir)

    for root, dnames, fnames in sorted(os.walk(root_dir, followlinks=recursively)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img_filepaths.append(path)

    if write_cache:
        filelist_cache = os.path.join(root_dir, 'files.list')
        with open(filelist_cache, 'w') as f:
            for path in img_filepaths:
                f.write("{}\n".format(path))
            print("Wrote filelist cache at {}".format(filelist_cache))

    return img_filepaths