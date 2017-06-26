import os
import pickle
import argparse
import skipthoughts
import sys

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--caption_file', type=str, default='Data/text.txt',
					   help='caption file')
	parser.add_argument('--data_dir', type=str, default='Data',
					   help='Data Directory')
	
	args = parser.parse_args()

	model = skipthoughts.load_model()
	encoded_captions = {}
	file_path = os.path.join(args.caption_file)
	dump_path = os.path.join(args.data_dir, 'enc_text.pkl')
	with open(file_path) as f:
		str_captions = f.read()
		captions = str_captions.split('\n')
		print(captions)
		encoded_captions['features'] = skipthoughts.encode(model, captions)

	pickle.dump(encoded_captions,
	            open(dump_path, "wb"))
	print('Finished extracting Skip-Thought vectors of the given text '
	      'descriptions')

if __name__ == '__main__':
	main()