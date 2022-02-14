import argparse
import numpy as np
import os

import pickle
import matlab
import matlab.engine


collist = matlab.double([0, 0.68, 0.7647])

def start_engine():
	isengine = matlab.engine.find_matlab()
	print(isengine)
	if not isengine:
		eng = matlab.engine.start_matlab()
		print("Starting matlab engine")
	else:
		eng = matlab.engine.connect_matlab(isengine[0])

	return eng

def convert_to_matlabint8(inarr):
	return matlab.int8(np.int8(np.ceil(inarr)).tolist())

def get_struct(folder, test_number, step=None):
	if step == None:
		file_name = []
		structs = []
		for path, currentDirectory, files in os.walk(folder):
			for file in files:
				if file.startswith(str(test_number)):
					print(path+"/"+file)

					with open(path+"/"+file, "rb+") as fp:
						struct = pickle.load(fp)
					structs.append(struct[0])
					file_name.append(path+"/"+file)
		return structs, file_name
	else:
		with open(desti, "rb+") as fp:
			struct = pickle.load(fp)[0]
		return struct, 0

def runstuff(folder, test_number, step=None):
	#desti = folder+"/"+test_number+"-"+step+".txt"
	struct, file_name = get_struct(folder, test_number, step)

	print(len(struct))
	eng = start_engine()
	#structog, vGextC, vGextF, vGstayOff = eng.get_struct2(nargout=4)
	#structog, _, vGextC, vGextF, vGstayOff = eng.get_struct3(nargout=5)
	structog, _, vGextC, vGextF, vGstayOff = eng.get_struct4(nargout=5)
	#structog, vGextC, vGextF, vGstayOff = eng.get_struct5(nargout=4)
	og_maxbending = eng.check_max_bend(structog, vGextC, vGextF, nargout=1)

	eng.clf(nargout=0)
	eng.plotVg_safe(structog, 'edgeOff', 'col',collist, nargout=0)
	if step == None:
		for i in range(len(struct)):
			print("Showing structure from ", file_name[i])
			structi = convert_to_matlabint8(struct[i])
			eng.clf(nargout=0)
			eng.plotVg_safe(structi, 'edgeOff', 'col',collist, nargout=0)
			try:
				new_bend = eng.check_max_bend(structi, vGextC, vGextF, nargout=1)
				if new_bend == 0 or np.isnan(new_bend):
					print("Doesn't work :P")
				else:
					print(new_bend, " / ", og_maxbending)
			except:
				print("Something went wrong :P")

			print('Press enter to close')
			input()
	else:
		print("Showing structure from " + desti)
		eng.clf(nargout=0)
		eng.plotVg_safe(convert_to_matlabint8(struct), 'edgeOff', 'col',collist, nargout=0)
		new_bend = eng.check_max_bend(convert_to_matlabint8(struct), vGextC, vGextF, nargout=1)
		print(og_maxbending, " / ", new_bend)
		print('Press enter to close')
		input()

def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_args():
	"""Parse command line argument."""
	parser = argparse.ArgumentParser("Train segmention model on 3D structures.")
	parser.add_argument("Folder", help="Folder with saved structures.")
	parser.add_argument("test_number", help="Get the structure from a specific run")
	parser.add_argument("checkpoint", type=none_or_str, nargs='?', default=None, help="Get the structure from a specific checkpoint")

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	runstuff(args.Folder, args.test_number, args.checkpoint)
