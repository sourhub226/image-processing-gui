import ctypes
import os

# import sys
from tkinter import *
from tkinter import filedialog, ttk
import cv2
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import scipy.signal

# np.set_printoptions(threshold=sys.maxsize)

ctypes.windll.shcore.SetProcessDpiAwareness(True)

root = Tk()
ttk.Style().configure("TButton", justify=CENTER)

# Global variables
gui_width = 1385
gui_height = 595
ip_file = ""
op_file = ""
original_img = None
modified_img = None
user_arg = None
popup = None
popup_input = None


root.title("Image processing")
# root.geometry(f"{gui_width}x{gui_height}")
# root.resizable(False, False)
root.minsize(gui_width, gui_height)


def set_user_arg():
    global user_arg
    user_arg = popup_input.get()
    popup.destroy()
    popup.quit()


def open_popup_input(text):
    global popup, popup_input
    popup = Toplevel(root)
    popup.resizable(False, False)
    popup.title("User Input")
    text_label = ttk.Label(popup, text=text, justify=LEFT)
    text_label.pack(side=TOP, anchor=W, padx=15, pady=10)
    popup_input = ttk.Entry(popup)
    popup_input.pack(side=TOP, anchor=NW, fill=X, padx=15)
    popup_btn = ttk.Button(popup, text="OK", command=set_user_arg).pack(pady=10)
    popup.geometry(f"400x{104+text_label.winfo_reqheight()}")
    popup_input.focus()
    popup.mainloop()


def draw_before_canvas():
    global original_img, ip_file
    original_img = Image.open(ip_file)
    original_img = original_img.convert("RGB")
    img = ImageTk.PhotoImage(original_img)
    before_canvas.create_image(
        256,
        256,
        image=img,
        anchor="center",
    )
    before_canvas.img = img


def draw_after_canvas(mimg):
    global modified_img

    modified_img = Image.fromarray(np.uint8(mimg))
    img = ImageTk.PhotoImage(modified_img)
    after_canvas.create_image(
        256,
        256,
        image=img,
        anchor="center",
    )
    after_canvas.img = img


def load_file():
    global ip_file
    ip_file = filedialog.askopenfilename(
        title="Open an image file",
        initialdir=".",
        filetypes=[("All Image Files", "*.*")],
    )
    draw_before_canvas()
    # print(f"Image loaded from: {ip_file}")


def save_file():
    global ip_file, original_img, modified_img
    file_ext = os.path.splitext(ip_file)[1][1:]
    op_file = filedialog.asksaveasfilename(
        filetypes=[
            (
                f"{file_ext.upper()}",
                f"*.{file_ext}",
            )
        ],
        defaultextension=[
            (
                f"{file_ext.upper()}",
                f"*.{file_ext}",
            )
        ],
    )
    modified_img = modified_img.convert("RGB")
    modified_img.save(op_file)
    # print(f"Image saved at: {op_file}")


# frames
left_frame = ttk.LabelFrame(root, text="Original Image", labelanchor=N)
left_frame.pack(fill=BOTH, side=LEFT, padx=10, pady=10, expand=1)

middle_frame = ttk.LabelFrame(root, text="Algorithms", labelanchor=N)
middle_frame.pack(fill=BOTH, side=LEFT, padx=5, pady=10)

right_frame = ttk.LabelFrame(root, text="Modified Image", labelanchor=N)
right_frame.pack(fill=BOTH, side=LEFT, padx=10, pady=10, expand=1)

# left frame contents
before_canvas = Canvas(left_frame, bg="white", width=512, height=512)
before_canvas.pack(expand=1)

browse_btn = ttk.Button(left_frame, text="Browse", command=load_file)
browse_btn.pack(expand=1, anchor=SW, pady=(5, 0))

# middle frame contents
algo_canvas = Canvas(middle_frame, width=260, highlightthickness=0)
scrollable_algo_frame = Frame(algo_canvas)
scrollbar = Scrollbar(
    middle_frame, orient="vertical", command=algo_canvas.yview, width=15
)
scrollbar.pack(side="right", fill="y")
algo_canvas.pack(fill=BOTH, expand=1)
algo_canvas.configure(yscrollcommand=scrollbar.set)
algo_canvas.create_window((0, 0), window=scrollable_algo_frame, anchor="nw")
scrollable_algo_frame.bind(
    "<Configure>", lambda _: algo_canvas.configure(scrollregion=algo_canvas.bbox("all"))
)


# right frame contents
after_canvas = Canvas(right_frame, bg="white", width=512, height=512)
after_canvas.pack(expand=1)

save_btn = ttk.Button(right_frame, text="Save", command=save_file)
save_btn.pack(expand=1, anchor=SE, pady=(5, 0))


# algorithm fns
def RGB2Gray():
    img = mpimg.imread(ip_file)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return 0.299 * R + 0.58 * G + 0.114 * B


def callRGB2Gray():
    grayscale = RGB2Gray()
    draw_after_canvas(grayscale)


def negative(set_gray):
    img = RGB2Gray() if (set_gray) else Image.open(ip_file)
    img = np.array(img)
    img = 255 - img
    draw_after_canvas(img)


def gray_slice(img, lower_limit, upper_limit, fn):
    # general function
    if lower_limit <= img <= upper_limit:
        return 255
    else:
        return fn


def call_gray_slice(retain):
    img = RGB2Gray()
    # input 100,180
    open_popup_input("Enter lower limit, upper limit\n(Separate inputs with a comma)")
    arg_list = user_arg.replace(" ", "").split(",")
    print(arg_list)
    lower_limit = int(arg_list[0])
    upper_limit = int(arg_list[1])
    img_thresh = np.vectorize(gray_slice)
    fn = img if retain else 0
    draw_after_canvas(img_thresh(img, lower_limit, upper_limit, fn))


def bit_slice(img, k):
    # create an image for the k bit plane
    plane = np.full((img.shape[0], img.shape[1]), 2 ** k, np.uint8)
    # execute bitwise and operation
    res = cv2.bitwise_and(plane, img)
    # multiply ones (bit plane sliced) with 255 just for better visualization
    return res * 255


def call_bit_slice():
    global user_arg
    bitplanes = []
    img = cv2.imread(ip_file, 0)
    open_popup_input(
        "Enter bit plane no k (0-7)\n(or leave it blank to display all 8 planes together)"
    )
    if not user_arg:
        for k in range(9):
            slice = bit_slice(img, k)
            # append to the output list
            slice = cv2.resize(slice, (171, 171))
            bitplanes.append(slice)

        # concat all 8 bit planes into one image
        row1 = cv2.hconcat([bitplanes[0], bitplanes[1], bitplanes[2]])
        row2 = cv2.hconcat([bitplanes[3], bitplanes[4], bitplanes[5]])
        row3 = cv2.hconcat([bitplanes[6], bitplanes[7], bitplanes[8]])
        final_img = cv2.vconcat([row1, row2, row3])
    else:
        final_img = bit_slice(img, int(user_arg))

    draw_after_canvas(final_img)


def c_stretch(img, r1, r2, s1, s2):
    # general function
    if img < r1:
        return s1
    elif img > r2:
        return s2
    else:
        return s1 + ((s2 - s1) * (img - r1) / (r2 - r1))


def call_c_stretch(limited):
    # input
    img = RGB2Gray()
    r1 = np.min(img)
    r2 = np.max(img)
    if limited:
        # input 25,220
        open_popup_input("Enter s1,s2\n(Separate inputs with a comma)")
        arg_list = user_arg.replace(" ", "").split(",")
        s1, s2 = int(arg_list[0]), int(arg_list[1])
    else:
        s1, s2 = (0, 255)
    image_cs = np.vectorize(c_stretch)
    draw_after_canvas(image_cs(img, r1, r2, s1, s2))


def plot_histogram(label, img, index):
    hist, bins = np.histogram(img, 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.subplot(1, 2, index)
    plt.title(label)
    plt.plot(cdf_normalized, color="b")
    plt.hist(img.flatten(), 256, [0, 256], color="r")
    plt.xlim([0, 256])
    plt.legend(("cdf", "histogram"), loc="upper left")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Distirbution")
    plt.tight_layout()


def histogram_eq():
    plt.figure(num=1, figsize=(11, 5), dpi=100)
    img = cv2.imread(ip_file, 0)
    plot_histogram("Original Histogram", img, 1)
    equ_img = cv2.equalizeHist(img)
    plot_histogram("Equalized Histogram", equ_img, 2)
    draw_after_canvas(equ_img)
    plt.show()


def correlate(image, filter):
    filtered_image = image
    for i in range(image.shape[-1]):
        filtered_image[:, :, i] = scipy.signal.correlate2d(
            image[:, :, i], filter, mode="same", boundary="symm"  # extended padding
        )
    filtered_image = filtered_image[:, :, ::-1]  # converts BGR to RGB
    return filtered_image


def box_filter():
    global user_arg
    open_popup_input("Enter n for (nxn) filter")
    user_arg = int(user_arg)
    filter = np.ones([user_arg, user_arg], dtype=int)
    filter = filter / (user_arg ** 2)
    image = cv2.imread(ip_file)
    filtered_image = correlate(image, filter)
    draw_after_canvas(filtered_image)


def wt_avg_filter():
    filter = [
        [1 / 16, 2 / 16, 1 / 16],
        [2 / 16, 4 / 16, 2 / 16],
        [1 / 16, 2 / 16, 1 / 16],
    ]
    image = cv2.imread(ip_file)
    filtered_image = correlate(image, filter)
    draw_after_canvas(filtered_image)


# algorithm btns
ttk.Button(
    scrollable_algo_frame, text="RGB to Grayscale", width=30, command=callRGB2Gray
).pack(expand=1, padx=5, pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Negative",
    width=30,
    command=lambda: negative(set_gray=False),
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Negative\n(Grayscale output)",
    width=30,
    command=lambda: negative(set_gray=True),
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Gray level slicing\n(retaining background)",
    width=30,
    command=lambda: call_gray_slice(retain=True),
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Gray level slicing\n(lowering background)",
    width=30,
    command=lambda: call_gray_slice(retain=False),
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Bit plane slicing",
    width=30,
    command=call_bit_slice,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Contrast Stretching\n(Linear)",
    width=30,
    command=lambda: call_c_stretch(limited=False),
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Contrast Stretching\n(Limited Linear)",
    width=30,
    command=lambda: call_c_stretch(limited=True),
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Histogram Equalization",
    width=30,
    command=histogram_eq,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Image Smoothing\n(nxn Avg/Box Filter)",
    width=30,
    command=box_filter,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Image Smoothing\n(3x3 Weighted Avg Filter)",
    width=30,
    command=wt_avg_filter,
).pack(pady=2, ipady=2)

# ttk.Button(
#     scrollable_algo_frame,
#     text="Image Smoothing\n(3x3 Median Filter)",
#     width=30,
#     command=wt_avg_filter,
# ).pack(pady=2, ipady=2)

# ttk.Button(
#     scrollable_algo_frame,
#     text="Image Smoothing\n(3x3 Weighted Median Filter)",
#     width=30,
#     command=wt_avg_filter,
# ).pack(pady=2, ipady=2)

root.mainloop()
