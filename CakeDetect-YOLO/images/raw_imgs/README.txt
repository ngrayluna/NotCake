How I got images:
1. Go to Google Images

2. Create Search

3. Go to View -> Developer -> Javascript Console

4. Go to Console

5. Scroll down until you have images you want.

6.Add the following code snippets to the Console:
// pull down jquery into the JavaScript console
var script = document.createElement('script');
script.src = "https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js";
document.getElementsByTagName('head')[0].appendChild(script);

//grab URLS
var urls = $('.rg_di .rg_meta').map(function() { return JSON.parse($(this).text()).ou; });

// write the URls to file (one per line)
var textToSave = urls.toArray().join('\n');
var hiddenElement = document.createElement('a');
hiddenElement.href = 'data:attachment/text,' + encodeURI(textToSave);
hiddenElement.target = '_blank';
hiddenElement.download = 'urls.txt';
hiddenElement.click();

This will (hopefully) download a .txt file with a list of the URLs.

7. Run download_images.py -u PATH -o OUTPATH


Note: You'll need the following libraries:
imutils
argparse
requests
cv2

Full tutorial can be found here:
https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/