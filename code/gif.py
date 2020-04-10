import os
import shutil
import imageio


def _make_meta_gifs():
    output_directory = "../output/gifs"
    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # Find how many versions have data
    for (dirpath, dirnames, filenames) in os.walk("../output"):
        n_versions = len(dirnames)
        # Ignore incomplete one
        n_versions -= 1
        break

    n_days = 100
    for version in range(n_versions):
        directory = f"../output/v{version}/plots"
        images = []
        for day in range(n_days):
            filename = os.path.join(directory, f'day{day}_iteration0.png')
            images.append(imageio.imread(filename))

        outfile = os.path.join("../output/gifs/", f"v{version}.gif")
        imageio.mimsave(outfile, images)

_make_meta_gifs()