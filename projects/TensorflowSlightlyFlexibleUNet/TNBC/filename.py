import os
import traceback

def get_name(filename):
  basename = os.path.basename(filename)
  
  splitted = os.path.splitext(filename)
  print("--- splitted {}".format(splitted))



if __name__ == "__main__":
  try:
    filename = "./distorted_0.01_rsigma0.5_sigma40_1015.jpg"
    get_name(filename)

  except:
    traceback.print_exc()

  