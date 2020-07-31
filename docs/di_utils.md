# di_utils.py

This is the Distributed Inference utility file, containing support utilities. 

#### read_labels

    read_labels(path)
    
Reads a labels file, where `path` is the path to the file. Returns a dictionary. 

For example, a list such as

    1 dog
    2 cat
    3 bird
    
would produce the dictionary

    {1: dog, 2: cat, 3: bird}
    
#### draw_labels

    draw_labels(
        img,
        proposals,
        labels=None
    )

Draws a label on the image (*function is in place, so it returns nothing*). `proposals` should be a list of proposals,
where each proposal is a 6-large array containing `[x1, y1, x2, y2, class, confidence]`. Labels are a dictionary or array
mapping numbers to text. Labels are optional, and if it is not specified the function will simply print the class ids. 