import imgsrcheng.inference
#del imgsrcheng.model

'''
Notes:
the above is used to remove unwanted modules from the namespace since I only
want to have access to the functions in inference. Python has this lovely
feature where if importing a module from the package requires access to anything
from another module in the package even if you are not importing the module directly,
Then those other modules are added to the overall package namespace. The above
solution seems a little hacky though.
'''
