import os
import sys
import numpy as np
import torch
import torch.nn as nn
import warnings
import argparse
import itertools

from overparam import OverparamLinear, OverparamConv2d


PASS_TOKEN = f'[\033[92mPASS\033[0m]'
FAIL_TOKEN = f'[\033[91mFAIL\033[0m]'


def GET_LINEAR_ARGUMENTS():
	arg_dicts = []
	arg_dicts += [{'param': 'batch_norm', 'args': [False, True]}]
	arg_dicts += [{'param': 'residual', 'args': [False, True]}]
	arg_dicts += [{'param': 'residual_intervals',
					'args': [1, 2, 4, -1, [1,2], [2,-1], [1,2,3], [1,2,4,-1]]}]
	arg_dicts += [{'param': 'bias', 'args': [False, True]}]
	arg_dicts += [{'param': 'depth', 'args': [1, 2, 4, 8, 16]}]
	arg_dicts += [{'param': 'width', 'args': [0.5, 1, 4]}]

	arguments = list(itertools.product(*[x['args'] for x in arg_dicts]))
	arg_names = [x['param'] for x in arg_dicts]
	return arguments, arg_names


def GET_CONV_ARGUMENTS():
	arg_dicts = []
	arg_dicts += [{'param': 'batch_norm', 'args': [False, True]}]
	arg_dicts += [{'param': 'residual', 'args': [False, True]}]
	arg_dicts += [{'param': 'residual_intervals',
					'args': [1, 2, 4, -1, [1,2], [2,-1], [1,2,3], [1,2,4,-1]]}]
	arg_dicts += [{'param': 'bias', 'args': [False, True]}]
	arg_dicts += [{'param': 'stride', 'args': [1, 2]}]
	arg_dicts += [{'param': 'kernel_sizes',
					'args': [1, 5, [3], [1, 3], [3, 1],
							 [5, 3, 1], [1, 3, 5], [3] * 8, [3] + 7 * [1]]}]
	arg_dicts += [{'param': 'width', 'args': [0.5, 1, 4]}]

	arguments = list(itertools.product(*[x['args'] for x in arg_dicts]))
	arg_names = [x['param'] for x in arg_dicts]
	return arguments, arg_names


def TEST_LINEAR_COMPUTATION():
	passed, total, failed = 0, 0, []

	arguments, arg_names = GET_LINEAR_ARGUMENTS()

	for args in arguments:
		torch.manual_seed(0)
		kwargs = {k:v for k, v in zip(arg_names, args)}

		if kwargs['residual']:
			if kwargs['width'] != 1:
				continue
		if not kwargs['residual']:
			if kwargs['residual_intervals'] is not 1:
				continue

		net = OverparamLinear(in_features=32, out_features=32, **kwargs).cuda().double()

		# random normal input
		x = torch.randn(16, 32).cuda().double()

		# train mode (expanded forward pass) [for warming up batch-norm]
		net.train()
		net(x)

		# eval mode (collapsed forward pass)
		net.eval()
		out1 = net(x)
		out2 = net(x, override=True)

		isclose = torch.allclose(out1, out2, atol=1e-6)

		msg_spec = '|'.join([f' {k}: {v} ' for k, v in kwargs.items()])
		msg_out = ' '.join([PASS_TOKEN if isclose else FAIL_TOKEN]) + msg_spec

		if not isclose:
			msg_out += f' (error: {(out1 - out2).abs().mean():.16f})'

		if verbose:
			print(msg_out)

		if not isclose:
			#net.visualize()
			input('failed.')
			pass

		if isclose:
			passed += 1

		total += 1

	if len(failed) > 0 and verbose:
		print('Failed runs ...')
		for x in failed:
			print(x)

	print(f'Passed [{passed}/{total}] tests')
	return passed == total


def TEST_CONV_COMPUTATION():
	passed, total, failed = 0, 0, []

	arguments, arg_names = GET_CONV_ARGUMENTS()

	for args in arguments:
		torch.manual_seed(0)
		kwargs = {k: v for k, v in zip(arg_names, args)}
		kwargs['padding'] = OverparamConv2d.compute_same_padding(kwargs['kernel_sizes'])

		if kwargs['residual']:
			if kwargs['width'] != 1:
				continue
		if not kwargs['residual']:
			if kwargs['residual_intervals'] is not 1:
				continue

		net = OverparamConv2d(in_channels=8, out_channels=8, **kwargs).double()

		# normal distribution N(0, I)
		x = torch.randn(1, 8, 32, 32).double()

		# train mode (expanded forward pass) [for warming up batch-norm]
		net.train()
		net(x)

		# eval mode (collapsed forward pass)
		net.eval()
		out1 = net(x)
		out2 = net(x, override=True)

		isclose = torch.allclose(out1, out2, atol=1e-6)

		msg_spec = '|'.join([f' {k}: {v} ' for k, v in kwargs.items()])
		msg_out = ' '.join([PASS_TOKEN if isclose else FAIL_TOKEN]) + msg_spec

		if not isclose:
			msg_out += f' (error: {(out1 - out2).abs().mean():.16f})'

		if verbose:
			print(msg_out)

		if not isclose:
			#net.visualize()
			input('failed.')
			pass

		if isclose:
			passed += 1

		total += 1

	if len(failed) > 0 and verbose:
		print('Failed runs ...')
		for x in failed:
			print(x)

	print(f'Passed [{passed}/{total}] tests')
	return passed == total


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='unittest')
	parser.add_argument('--ALL', action='store_true')
	parser.add_argument('--LINEAR', action='store_true')
	parser.add_argument('--CONV', action='store_true')
	args = parser.parse_args()

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit()

	verbose = True

	if not verbose:
		warnings.filterwarnings("ignore")

	TEST_RESULTS = []

	if args.ALL or args.LINEAR:
		print('> TESTING `OverparamLinear` COMPUTATION CONSISTENCY')
		SUCCESS = TEST_LINEAR_COMPUTATION()
		TEST_RESULTS += [['TEST_LINEAR_COMPUTATION', SUCCESS]]

	if args.ALL or args.CONV:
		print('> TESTING `OverparamConv2d` COMPUTATION CONSISTENCY')
		SUCCESS = TEST_CONV_COMPUTATION()
		TEST_RESULTS += [['TEST_CONV_COMPUTATION', SUCCESS]]


	print('=' * 20)
	for test_name, all_passed in TEST_RESULTS:
		print(f'>> {test_name} {PASS_TOKEN if all_passed else FAIL_TOKEN}')
