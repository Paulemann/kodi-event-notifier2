#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Install opencv with;
# pip3 install opencv-python
#

import time
from datetime import datetime
import requests
import json

import logging
import configparser
import os
import sys
import socket
import signal
import subprocess

import email
from email.utils import formataddr
from email.header import Header, decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
import smtplib

import argparse

import base64
import asyncore
from smtpd import SMTPServer, SMTPChannel, DEBUGSTREAM

import numpy as np
import cv2
from imutils import paths

import re

# global settings
_max_waittime_ = 10
_padding_ = ' ' * 11

# Pretrained classes in the model
classNames = {0: 'Background',
              1: 'Person', 2: 'Bicycle', 3: 'Car', 4: 'Motorcycle', 5: 'Airplane', 6: 'Bus',
              7: 'Train', 8: 'Truck', 9: 'Boat', 10: 'Traffic light', 11: 'Fire Hydrant',
              13: 'Stop Sign', 14: 'Parking Meter', 15: 'Bench', 16: 'Bird', 17: 'Cat',
              18: 'Dog', 19: 'Horse', 20: 'Sheep', 21: 'Cow', 22: 'Elephant', 23: 'Bear',
              24: 'Zebra', 25: 'Giraffe', 27: 'Backpack', 28: 'Umbrella', 31: 'Handbag',
              32: 'Tie', 33: 'Suitcase', 34: 'Frisbee', 35: 'Skis', 36: 'Snowboard',
              37: 'Sports Ball', 38: 'Kite', 39: 'Baseball Bat', 40: 'Baseball Glove',
              41: 'Skateboard', 42: 'Surfboard', 43: 'Tennis Racket', 44: 'Bottle',
              46: 'Wine Glass', 47: 'Cup', 48: 'Fork', 49: 'Knife', 50: 'Spoon',
              51: 'Bowl', 52: 'Banana', 53: 'Apple', 54: 'Sandwich', 55: 'Orange',
              56: 'Broccoli', 57: 'Carrot', 58: 'Hot Dog', 59: 'Pizza', 60: 'Donut',
              61: 'Cake', 62: 'Chair', 63: 'Couch', 64: 'Potted Plant', 65: 'Bed',
              67: 'Dining Table', 70: 'Toilet', 72: 'TV', 73: 'Laptop', 74: 'Mouse',
              75: 'Remote', 76: 'Keyboard', 77: 'Cell Phone', 78: 'Microwave', 79: 'Oven',
              80: 'Toaster', 81: 'Sink', 82: 'Refrigerator', 84: 'Book', 85: 'Clock',
              86: 'Vase', 87: 'Scissors', 88: 'Teddy Bear', 89: 'Hair Drier', 90: 'Toothbrush'}

matchObject = 'Person'

def id_class_name(class_id, classes):
        for key,value in classes.items():
                if class_id == key:
                        return value


def detect(image, cropX, cropY, cropW, cropH, threshold=0.5, highlight=False, matchobject=None):
  # crop the input image and construct an input blob for the image
  # by resizing to a fixed 300x300 pixels and then normalizing it

  if cropW  > 0 and cropH > 0:
    if cropW != cropH:
      size = min(cropW, cropH)
      cropW = size
      cropH = size
    log('Cropping input image to focus area (x, y, w, h): {}, {}, {}, {} ...'.format(cropX, cropY, cropW, cropH), level='DEBUG')
  else:
    (h, w) = image.shape[:2]
    size = min(w, h)
    cropX = int((w - size) / 2)
    cropY = int((h - size) / 2)
    cropW = size
    cropH = size
    log('Cropping input image to square format (x, y, w, h): {}, {}, {}, {} ...'.format(cropX, cropY, cropW, cropH), level='DEBUG')

  cropped_image = image[cropY: cropY + cropH, cropX: cropX + cropW].copy()
  (h, w) = cropped_image.shape[:2] # or use cropH, cropW

  if _use_caffe_:
    blob = cv2.dnn.blobFromImage(cropped_image, scalefactor=1.0/127.5, size=(300, 300), mean=(127.5, 127.5, 127.5), swapRB=False, crop=False)
  else:
    blob = cv2.dnn.blobFromImage(cropped_image, scalefactor=1.0, size=(300, 300), swapRB=True, crop=False)

  # pass the blob through the network and obtain the detections and predictions
  _net_.setInput(blob)
  detections = _net_.forward()

  found = 0
  # loop over the detections
  for detection in detections[0, 0]:
    # extract the confidence score (i.e., probability) associated with the prediction
    score = detection[2]

    # filter out weak detections by ensuring the 'score' is greater than the confidence threshold
    if score > threshold:
      if highlight or matchobject:
        if not _cv_model_: # we're using the default model with known classNames
          object = id_class_name(detection[1], classNames)
          if matchobject and object != matchobject:
            continue
          text = '{}: {:.2f}%'.format(object, score * 100)
        else:
          text = '{:.2f}%'.format(score * 100)

        if highlight:
          startX = int(detection[3] * w + cropX)
          startY = int(detection[4] * h + cropY)
          endX   = int(detection[5] * w + cropX)
          endY   = int(detection[6] * h + cropY)

          # draw the bounding box of the object
          cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

          (tw, th) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)[0]

          #draw filled rectangle as text background
          cv2.rectangle(image, (startX - 1, startY - th - 8), (startX + tw, startY - 2), (0, 255, 0), cv2.FILLED)

          cv2.putText(image, text, (startX, startY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

      found += 1

  return found


def is_mailaddress(a):
  try:
    t = a.split('@')[1].split('.')[1]
  except:
    return False

  return True


def is_hostname(h):
  try:
    t = h.split('.')[2]
  except:
    return False

  return True


def is_int(n):
  try:
    t = int(n)
  except:
    return False

  return True


def log(message, level='INFO'):
  if _log_file_:
    if level == 'DEBUG' and _debug_:
      logging.debug(message)
    if level == 'INFO':
      logging.info(message)
    if level == 'WARNING':
      logging.warning(message)
    if level == 'ERROR':
      logging.error(message)
    if level == 'CRITICAL':
      logging.crtitcal(message)
  else:
     if level != 'DEBUG' or _debug_:
       print('[{:^8}] '.format(level) + message)


def read_config():
  global _kodi_host_, _kodi_port_, _kodi_user_, _kodi_passwd_
  global _local_user_, _local_passwd_, _local_port_
  global _event_str_, _event_id_
  global _attach_exec_, _attach_analyze_, _attach_path_
  global _device_str_, _device_id_, _device_description_, _device_stream_id_, _device_croparea_
  global _smtp_server_, _smtp_port_, _smtp_realname_, _smtp_user_, _smtp_passwd_
  global _mail_to_, _mail_subject_, _mail_to_replace_, _mail_subject_replace_,  _mail_body_, _time_fmt_
  global _notify_title_, _notify_text_

  if not os.path.exists(_config_file_):
    log('Could not find configuration file \'{}\'.'.format(_config_file_), level='ERROR')
    return False

  log('Reading configuration from file {} ...'.format(_config_file_))

  config = configparser.ConfigParser(interpolation=None)
  config.read([os.path.abspath(_config_file_)], encoding='utf-8')
  try:
    # Read the config file
    #config = configparser.ConfigParser(interpolation=None)
    #config.read([os.path.abspath(_config_file_)], encoding='utf-8')

    _kodi_user_   = config.get('Kodi', 'username').strip(' "\'')
    _kodi_passwd_ = config.get('Kodi', 'password').strip(' "\'')

    _kodi_port_   = int(config.get('Kodi', 'port').strip(' "\''))
    if not is_int(_kodi_port_):
      log('{}: Invalid value of parameter \'{}\' in section [{}].'.format(_config_file_, 'port', 'Kodi'), log='ERROR')
      return False

    _kodi_host_  = [p.strip(' "\'') for p in config.get('Event', 'notify').split(',')]
    for host in _kodi_host_:
      if not is_hostname(host):
        log('{}: Invalid value of parameter \'{}\' in section [{}].'.format(_config_file_, 'notify', 'Event'), log='ERROR')
        return False

    _event_str_  = config.get('Event', 'searchfor').strip(' "\'')
    #_event_id_   = config.get('Event', 'eventid').strip(' "\'')
    # Allow multiple events as trigger
    _event_id_   = [p.strip(' "\'') for p in config.get('Event', 'eventid').split(',')]

    _attach_exec_    = config.get('Attachments', 'generate').strip(' "\'')

    _attach_analyze_ = config.get('Attachments', 'analyze').strip(' "\'').lower()
    if _attach_analyze_ == 'true' or _event_analyze_ == 'yes':
      _attach_analyze_ = True
    else:
      _attach_analyze_ = False

    _attach_path_ = [p.strip() for p in config.get('Attachments', 'searchpath').split(',')]
    if _attach_path_ == ['']:
      _attach_path_ = None

    _mail_to_ = [p.strip(' "\'').lower() for p in config.get('Event', 'sendmessage').split(',')]

    try:
      _device_str_ = config.get('Event', 'searchfrom').strip(' "\'')
    except:
      _device_str_ = ''

    if not _device_str_:
      log('{}: Value of parameter \'{}\' in section [{}] not set. Will search \'from\' field of event message for device info.'.format(_config_file_, 'searchfrom', 'Event'), log='ERROR')

    _device_id_          = [p.strip(' "\'') for p in config.get('Device', 'deviceid').split(',')]
    _device_description_ = [p.strip(' "\'') for p in config.get('Device', 'description').split(',')]

    try:
      _device_stream_id_ = [int(p.strip(' "\'')) for p in config.get('Device', 'streamid').split(',')]
    except:
      _device_stream_id_ = []

    if _device_stream_id_:
      for id in _device_stream_id_:
        if not is_int(id):
          log('{}: Invalid value of parameter \'{}\' in section [{}].'.format(_config_file_, 'streamid', 'Device'), log='ERROR')
          return False

    try:
      # read _device_croparea_ as a list of tuples of type (x,y,w,h)
      l = [int(p.strip()) for p in config.get('Device', 'focusarea').split(',')]
      _device_croparea_ = [(l[i], l[i+1], l[i+2], l[i+3]) for i in range(0, len(l), 4)]
    except:
      _device_croparea_ = []

    _local_port_ = int(config.get('Local Server', 'port').strip(' "\''))
    if not is_int(_local_port_):
      log('{}: Invalid value of parameter \'{}\' in section [{}].'.format(_config_file_, 'port', 'Local Server'), log='ERROR')
      return False

    _local_user_   = config.get('Local Server', 'username').strip(' "\'')
    _local_passwd_ = config.get('Local Server', 'password').strip(' "\'')

    try:
      _smtp_server_ = config.get('SMTP Server', 'hostname').strip(' "\'')
    except:
      _smtp_server_ = ''

    if not _mail_to_[0] or _mail_to_[0] == 'false' or _mail_to_[0] == 'no':
      log('{}: Value of parameter \'{}\' in section [{}]: \'{}\'. Will not forward messages.'.format(_config_file_, 'sendmessage', 'Event', _mail_to_[0]), log='DEBUG')
      _mail_to_     = []
      _smtp_server_ = ''

    if _smtp_server_ and is_hostname(_smtp_server_):
      _smtp_port_ = int(config.get('SMTP Server', 'port').strip(' "\''))
      if not is_int(_smtp_port_):
        log('{}: Invalid value of parameter \'{}\' in section [{}].'.format(_config_file_, 'port', 'SMTP Server'), log='ERROR')
        return False
      _smtp_user_ = config.get('SMTP Server', 'username').strip(' "\'')
      if not is_mailaddress(_smtp_user_):
        log('{}: Invalid value of parameter \'{}\' in section [{}].'.format(_config_file_, 'username', 'SMTP Server'), log='ERROR')
        return False

      _smtp_realname_ = config.get('SMTP Server', 'realname').strip(' "\'')
      _smtp_passwd_   = config.get('SMTP Server', 'password').strip(' "\'')

      _mail_to_replace_ = True
      for addr in _mail_to_:
        if addr == 'true' or addr == 'yes' or not is_mailaddress(addr):
          log('{}: Value of parameter \'{}\' in section [{}]: \'{}\'. Will use recipient list of original message.'.format(_config_file_, 'sendmessage', 'Event', addr), level='DEBUG')
          _mail_to_         = []
          _mail_to_replace_ = False
          break

      try:
        _mail_subject_         = config.get('Event Message', 'subject').strip(' "\'')
        _mail_subject_replace_ = True
      except:
        _mail_subject_         = ''

      if not _mail_subject_:
        _mail_subject_replace_ = False
        log('{}: Value of parameter \'{}\' in section [{}] not set. Will use subject of original message.'.format(_config_file_, 'subject', 'Event Message'), level='DEBUG')

      _mail_body_ = config.get('Event Message', 'text').strip(' "\'').replace('\\n', '\r\n')
      if not _mail_body_:
        log('{}: Missing value of parameter \'{}\' in section [{}].'.format(_config_file_, 'text', 'Event Message'), log='ERROR')
        return False

      _time_fmt_    = config.get('Event Message', 'timeformat').strip(' "\'')

      if not _time_fmt_:
        _time_fmt_ = "%Y-%m-%d %H:%M:%S"
    else:
      log('{}: Invalid value of parameter \'{}\' in section [{}].'.format(_config_file_, 'hostname', 'SMTP Server'))
      return False

    _notify_title_ = config.get('Event Notification', 'title').strip(' "\'')
    _notify_text_  = config.get('Event Notification', 'text').strip(' "\'')

  except:
    log('Could not process configuration file.', level='ERROR')
    return False

  log('Configuration OK.')

  return True


def kodi_request(host, method, params):
  url  = 'http://{}:{}/jsonrpc'.format(host, _kodi_port_)
  headers = {'content-type': 'application/json'}
  data = {'jsonrpc': '2.0', 'method': method, 'params': params,'id': 1}

  if _kodi_user_ and _kodi_passwd_:
    base64str = base64.encodestring('{}:{}'.format(_kodi_user_, _kodi_passwd_))[:-1]
    header['Authorization'] = 'Basic {}'.format(base64str)

  try:
    response = requests.post(url, data=json.dumps(data), headers=headers, timeout=10)
  except:
    return False

  data = response.json()
  return (data['result'] == 'OK')


def host_is_up(host, port):
  try:
    sock = socket.create_connection((host, port), timeout=3)
  #except socket.timout:
  #  return False
  except:
    return False

  return True


def send_msg(device_id, attachments=None):
  #
  # https://code.tutsplus.com/tutorials/sending-emails-in-python-with-smtp--cms-29975
  #

  # In case of test, recipients and subject may be empty
  if not _mail_to_:
    recipients = [_smtp_user_]
  else:
    recipients = _mail_to_

  device_index = _device_id_.index(device_id)

  try:
    subject = _mail_subject_.format(_device_description_[device_index])
  except:
    subject = _mail_subject_

  if not subject:
    subject = 'Test'

  try:
    now = datetime.now().strftime(_time_fmt_)
    body = '{}: '.format(now) + _mail_body_.format(_device_description_[device_index])
  except:
    body = _mail_body_

  if not body:
    log('Cannot send message. Missing message data.', level='ERROR')
    return False

  files = attachments
  if _attach_path_ and len(_attach_path_) == len(_device_id_):
    index = device_index
  else:
    index = 0
  if not files and _attach_path_ and os.path.isdir(_attach_path_[index]):
    p = _attach_path_[index]
    log('Searching directory for attachment(s): {} ...'.format(p), level='DEBUG')

    waittime = 0
    while not next(os.walk(p))[2] and waittime < _max_waittime_:
      waittime += 1
      time.sleep(1)

    files = [(os.path.join(p, f), None) for f in sorted(os.listdir(p)) if os.path.isfile(os.path.join(p, f))]
    log('Found {} file(s) to attach.'.format(len(files)), level='DEBUG')
  elif files:
    log('Forwarding {} attachment(s) from original message ...'.format(len(files)), level='DEBUG')

  if len(_device_croparea_) > device_index:
    (cropX, cropY, cropW, cropH) = _device_croparea_[device_index]
  else:
    (cropX, cropY, cropW, cropH) = (0, 0, 0, 0)

  log('Sending message via {} ...'.format(_smtp_server_), level='DEBUG')

  msg = MIMEMultipart()

  if _smtp_realname_:
    msg['From']  = formataddr((str(Header(_smtp_realname_, 'utf-8')), _smtp_user_))
  else:
    msg['From']  = _smtp_user_
  msg['To']      = ', '.join(recipients)
  msg['Subject'] = subject

  log('Assembling message with subject \'{}\' ...'.format(msg['Subject']), level='DEBUG')

  msg.attach(MIMEText(body, 'plain'))

  objects_detected = 0
  for name, content in files or []:
    log('Processing attachment: {} ...'.format(name), level='DEBUG')
    try:
      if not content and os.path.isfile(name):
        with open(name, 'rb') as f:
          payload = f.read()
      elif content:
        payload = content
      else:
        continue
      if _attach_analyze_:
        num_objects, image = analyze(payload, cropX, cropY, cropW, cropH, os.path.splitext(name)[1], matchobject=matchObject)
        if num_objects > 0:
          if num_objects > objects_detected:
            objects_detected = num_objects
          if image:
            payload = image
      part = MIMEBase('application', "octet-stream")
      part.set_payload(payload)
      email.encoders.encode_base64(part)
      part.add_header('Content-Disposition', 'attachment; filename="{}"'.format(os.path.basename(name)))
      msg.attach(part)
    except Exception as e:
      log('Couldn\'t process attachment: \'{}\'. Proceeding ...'.format(e), level='ERROR')
      continue

  if _attach_analyze_:
    if objects_detected > 0:
      log('Maximum of {} object(s) detected in attachments. Proceeding ...'.format(objects_detected))
    else:
      log('No objects detected in attachment(s). Skip forwarding message.')
      return False

  try:
    with smtplib.SMTP(_smtp_server_, _smtp_port_) as server:
      server.starttls()
      server.login(_smtp_user_, _smtp_passwd_)
      server.sendmail(_smtp_user_, msg['To'].split(','), msg.as_string())

  except Exception as e:
    log('Failed to send message: \'{}\'.'.format(e), level='ERROR')
    return False

  #finally:
  #  server.quit()

  log('Message successfully sent to recipient(s): {}.'.format(msg['To']))
  return True


def alert(device_id, attachments=None):
  #device_index = _device_id_.index(device_id)

  # This will execute the configured local command passing the device id as add. argument
  # Attention: Script waits for command to terminate and return
  if not attachments and _attach_exec_:
    try:
      log('Executing local command: {} {} ...'.format(_attach_exec_, device_id), level='DEBUG')
      parms = _attach_exec_.split()
      parms.append(device_id)
      subprocess.call(parms)
    except Exception as e:
      log('Excution failed with exception: \'{}\'. Proceeding ...'.format(e), level='ERROR')
      pass

  for host in _kodi_host_:
    log('Initiating communication with kodi host: {} ...'.format(host))

    if not host_is_up(host, _kodi_port_):
      log('Host is down. Action canceled.')
      continue

    notify(host, device_id)

  if _smtp_server_:
    send_msg(device_id, attachments=attachments)


def notify(host, device_id):
  if _notify_title_ and _notify_text_:
    device_index = _device_id_.index(device_id)

    try:
      text = _notify_text_.format(_device_description_[device_index])
    except:
      text = _notify_text_

    log('Sending notification \'{}: {}\' ...'.format(_notify_title_, text), level='DEBUG')
    kodi_request(host, 'GUI.ShowNotification', {'title': _notify_title_, 'message': text, 'displaytime': 2000})

  if _addon_id_:
    if _device_stream_id_ and len(_device_stream_id_) == len(_device_id_) and device_id:
      stream_id = _device_stream_id_[device_index]
    else:
      stream_id = 0

    log('Calling addon \'{}\' for stream id {} ...'.format(_addon_id_, stream_id), level='DEBUG')
    kodi_request(host, 'Addons.ExecuteAddon', {'addonid': _addon_id_, 'params': {'streamid': str(stream_id)}})


def analyze(content, cropX, cropY, cropW, cropH, ext, matchobject=None):
  enc_image = None

  image = cv2.imdecode(np.fromstring(content, dtype=np.uint8), -1)
  #image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), flags=1)
  detected = detect(image, cropX, cropY, cropW, cropH, threshold=_threshold_, highlight=True, matchobject=matchobject)
  log('{} Object(s) detetced.'.format(detected), level='DEBUG')

  if detected > 0:
    enc_image = cv2.imencode(ext, image)[1].tostring()
    #enc_iamge = cv2.imencode(ext, image)[1].tobytes()

  return detected, enc_image

def decode_b64(data):
  """Wrapper for b64decode, without having to struggle with bytestrings."""
  byte_string = data.encode('utf-8')
  decoded = base64.b64decode(byte_string)
  return decoded.decode('utf-8')


def encode_b64(data):
  """Wrapper for b64encode, without having to struggle with bytestrings."""
  byte_string = data.encode('utf-8')
  encoded = base64.b64encode(byte_string)
  return encoded.decode('utf-8')


class FakeCredentialValidator(object):
  def __init__(self, username, password, channel):
    self.username = username
    self.password = password
    self.channel = channel

  def validate(self):
    log('Receiving authentication request for user: {} ...'.format(self.username), level='DEBUG')

    if self.username == _local_user_ and self.password == _local_passwd_:
      log('Authentication successful.', level='DEBUG')
      return True

    log('Authentication failed.', level='ERROR')
    return False


class MySMTPChannel(SMTPChannel):
  credential_validator = FakeCredentialValidator

  def __init__(self, server, conn, addr, *args, **kwargs):
    super().__init__(server, conn, addr, *args, **kwargs)
    self.username = None
    self.password = None
    self.authenticated = False
    self.authenticating = False

  def smtp_AUTH(self, arg):
    if 'PLAIN' in arg:
      split_args = arg.split(' ')
      # second arg is Base64-encoded string of blah\0username\0password
      authbits = decode_b64(split_args[1]).split('\0')
      self.username = authbits[1]
      self.password = authbits[2]
      if self.credential_validator and self.credential_validator(self.username, self.password, self).validate():
        self.authenticated = True
        self.push('235 Authentication successful.')
      else:
        self.push('454 Temporary authentication failure.')
        self.close_when_done()

    elif 'LOGIN' in arg:
      self.authenticating = True
      split_args = arg.split(' ')

      # Some implmentations of 'LOGIN' seem to provide the username
      # along with the 'LOGIN' stanza, hence both situations are
      # handled.
      if len(split_args) == 2:
        self.username = decode_b64(arg.split(' ')[1])
        self.push('334 ' + encode_b64('Username'))
      else:
        self.push('334 ' + encode_b64('Username'))

    elif not self.username:
      self.username = decode_b64(arg)
      self.push('334 ' + encode_b64('Password'))

    else:
      self.authenticating = False
      self.password = decode_b64(arg)
      if self.credential_validator and self.credential_validator(self.username, self.password, self).validate():
        self.authenticated = True
        self.push('235 Authentication successful.')
      else:
        self.push('454 Temporary authentication failure.')
        self.close_when_done()

  def smtp_EHLO(self, arg):
    if not arg:
      self.push('501 Syntax: EHLO hostname')
      return
    if self.seen_greeting:
      self.push('503 Duplicate HELO/EHLO')
      return
    self._set_rset_state()
    self.seen_greeting = arg
    self.extended_smtp = True
    self.push('250-{}'.format(self.fqdn))
    self.push('250-AUTH LOGIN PLAIN')
    self.push('250-AUTH LOGIN PLAIN')
    if self.data_size_limit:
      self.push('250-SIZE {}'.format(self.data_size_limit))
      self.command_size_limits['MAIL'] += 26
    if not self._decode_data:
      self.push('250-8BITMIME')
    if self.enable_SMTPUTF8:
      self.push('250-SMTPUTF8')
      self.command_size_limits['MAIL'] += 10
    self.push('250 HELP')

  def smtp_HELO(self, arg):
    if not arg:
      self.push('501 Syntax: HELO hostname')
      return
    if self.seen_greeting:
      self.push('503 Duplicate HELO/EHLO')
      return
    self._set_rset_state()
    self.seen_greeting = arg
    self.push('250 {}'.format(self.fqdn))

  def run_command_with_arg(self, command, arg):
    method = getattr(self, 'smtp_' + command, None)
    if not method:
       self.push('500 Error: command "{}" not recognized'.format(command))
       return

    # White list of operations that are allowed prior to AUTH.
    if command not in ['AUTH', 'EHLO', 'HELO', 'NOOP', 'RSET', 'QUIT']:
      if not self.authenticated:
        self.push('530 Authentication required')
        return

    method(arg)

  def found_terminator(self):
    line = self._emptystring.join(self.received_lines)
    print('Data:', repr(line), file=DEBUGSTREAM)
    self.received_lines = []
    if self.smtp_state == self.COMMAND:
      sz, self.num_bytes = self.num_bytes, 0
      if not line:
        self.push('500 Error: bad syntax')
        return
      if not self._decode_data:
        line = str(line, 'utf-8')
      i = line.find(' ')

      if self.authenticating:
        # If we are in an authenticating state, call the
        # method smtp_AUTH.
        arg = line.strip()
        command = 'AUTH'
      elif i < 0:
        command = line.upper()
        arg = None
      else:
        command = line[:i].upper()
        arg = line[i + 1:].strip()
      max_sz = (self.command_size_limits[command] if self.extended_smtp else self.command_size_limit)

      if sz > max_sz:
        self.push('500 Error: line too long')
        return

      self.run_command_with_arg(command, arg)
      return
    else:
      if self.smtp_state != self.DATA:
        self.push('451 Internal confusion')
        self.num_bytes = 0
        return
      if self.data_size_limit and self.num_bytes > self.data_size_limit:
        self.push('552 Error: Too much mail data')
        self.num_bytes = 0
        return
      # Remove extraneous carriage returns and de-transparency according
      # to RFC 5321, Section 4.5.2.
      data = []
      for text in line.split(self._linesep):
        if text and text[0] == self._dotsep:
          data.append(text[1:])
        else:
          data.append(text)
      self.received_data = self._newline.join(data)
      args = (self.peer, self.mailfrom, self.rcpttos, self.received_data)
      kwargs = {}
      if not self._decode_data:
        kwargs = {
          'mail_options': self.mail_options,
          'rcpt_options': self.rcpt_options,
        }
      status = self.smtp_server.process_message(*args, **kwargs)
      self._set_post_data_state()
      if not status:
        self.push('250 OK')
      else:
        self.push(status)


class MySMTPServer(SMTPServer):
  channel_class = MySMTPChannel

  def process_message(self, peer, mailfrom, rcpttos, data):
    global _mail_to_, _mail_subject_

    try:
      #Implement additional security checks here: e.g. filter on mailfrom and/or addr.
      addr, port = peer
      log('Receiving message from: {}:{}'.format(addr, port))
      log('Message sent from:      {}'.format(mailfrom))
      log('Message addressed to:   {}'.format(', '.join(rcpttos)))

      msg = email.message_from_string(data)
      subject = ''
      for encoded_string, charset in decode_header(msg.get('Subject')):
        try:
          if charset is not None:
            subject += encoded_string.decode(charset)
          else:
            subject += encoded_string
        except:
          log('Error reading part of subject: {} charset {}'.format(encoded_string, charset))
      log('Message subject:        {}'.format(subject))

      if not _mail_to_replace_: #Maintain recipient list from original message
        _mail_to_ = rcpttos
      if not _mail_subject_replace_: #Maintain subject from original message
        _mail_subject_ = subject

      headers = '\n'.join((_padding_ + str(key) + ': ' + str(val)) for key, val in msg.items())
      log('Message headers:\n' + headers, level='DEBUG')

      text_parts = []
      attachments = []

      # loop on the email parts
      for part in msg.walk():
        if part.get_content_maintype() == 'multipart':
          continue

        c_type = part.get_content_type()
        c_disp = part.get('Content-Disposition')

        # text parts will be appended to text_parts
        if c_type == 'text/plain' and c_disp == None:
          text_parts.append(part.get_payload(decode=True).decode('utf-8').strip())
        # ignore html part
        elif c_type == 'text/html':
          continue
        # attachments will be sent as files in the POST request
        else:
          filename = part.get_filename()
          filecontent = part.get_payload(decode=True)
          if filecontent is not None:
            if filename is None:
              filename = 'untitled{}'.format(len(attachments))
            attachments.append((filename, filecontent))
            log('Message attachment: file{} = {}'.format(len(attachments), filename), level='DEBUG')

      mailbody = '\r\n'.join(text_parts)

      lines = '\n'.join([(_padding_ + l.strip()) for l in mailbody.split('\r')])
      log('Message body:\n' + lines, level='DEBUG')

      if len(lines.splitlines()) == 1: # Foscam
        log('Looks like a foscam cam ...', level='DEBUG')

        mailbody = ''
        if _device_str_:
          p1 = 'Your IPCamera:'
          r1 = _device_str_ + ':'
          mailbody += re.sub(r'.*(?<=' + p1 + ')\s*(\S*).*', r1 + r' \1', lines) + '\r\n'

          #p2 = 'IP:'
          #r2 = 'IP Address:'
          #mailbody += re.sub(r'.*(?<=' + p2 + ')\s*(\S*).*', r2 + r' \1', lines) + '\r\n'

        if _event_str_:
          p3 = 'detected'
          p4 = 'at'
          r3 = _event_str_ + ':'
          mailbody += re.sub(r'.*(?<=' + p3 + ') (.*) ' + p4 + ' .*$', r3 + r' \1', lines)

        lines = '\n'.join([(_padding_ + l.strip()) for l in mailbody.split('\r')])
        log('Adapted message body:\n' + lines, level='DEBUG')

    except:
      log('Error reading incoming message', level='ERROR')

    validate(mailfrom, mailbody, attachments)


def validate(msgfrom, msgbody, attachments):
  event_data =  {}
  for line in [l.strip() for l in msgbody.split('\r')]:
    line = line.replace(': ', '= ')
    args = [p.strip() for p in line.rsplit('=', 1)]
    if len(args) == 2:
      event_data[args[0]] = args[1]

  if not _event_str_ or not _event_id_:
    log("No event configured for processing.", level='ERROR')
    return

  if _event_str_ not in event_data:
    log("No data to identify this event.", level='ERROR')
    return

  if _device_str_ and _device_id_:
    log("Searching \'body\' of event message for device info ...", level='DEBUG')
    if _device_str_ not in event_data:
      log("No data to identify this device.", level='ERROR')
      return
    log("Found device info: {}".format(event_data[_device_str_]), level='DEBUG')
    if event_data[_device_str_] not in _device_id_:
      log("Not processing events from this device.")
      return
    device_id = event_data[_device_str_]
  elif _device_id_: # look in from data of sent message
    log("Using device info in\'from\' field of event message: {}".format(msgfrom), level='DEBUG')
    if msgfrom not in device_id:
      log("Not processing events from this device.")
      return
    device_id = msgfrom

  if event_data[_event_str_] in _event_id_:
    log("Message has alarm event: {}. Processing ...".format(event_data[_event_str_]))
    alert(device_id, attachments=attachments)
  else:
    log("Message has alarm event: {}. This event is not processed. ".format(event_data[_event_str_]))


def port_is_used(port):
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    return s.connect_ex(('localhost', port)) == 0


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Sends a notification to a kodi host and triggers addon execution on receipt of an external 433 MHz signal')

  parser.add_argument('-d', '--debug', dest='debug', action='store_true', help="Output debug messages (Default: False)")
  parser.add_argument('-l', '--logfile', dest='log_file', default=None, help="Path to log file (Default: None=stdout)")
  parser.add_argument('-c', '--config', dest='config_file', default=os.path.splitext(os.path.basename(__file__))[0] + '.ini', help="Path to config file (Default: <Script Name>.ini)")
  parser.add_argument('-a', '--addonid', dest='addon_id', default='script.securitycam', help="Addon ID (Default: script.securitycam)")
  parser.add_argument('-t', '--test', dest='test', action='store_true', help="Test with simulated event (Default: False)")
  parser.add_argument('-w', '--weights', dest='cv_model', default=None, help="Overwrites path to OpenCV's trained weights binary file (Ext.: .caffemodel or .pb)")
  parser.add_argument('-n', '--netconfig', dest='cv_config', default=None, help="Overwrites path to OpenCV's network config file (Ext.: .prototxt or .pbtxt)")
  parser.add_argument('-u', '--usecaffe', dest='use_caffe', action='store_true', help="Use Caffe method instead of default TensorFlow method (Default: False)")
  parser.add_argument('-f', '--threshold', dest='threshold', type=float, default=0.6, help="Probability threshold to filter weak detections (Default: 0.6)")

  args = parser.parse_args()

  _config_file_ = args.config_file
  _log_file_    = args.log_file
  _addon_id_    = args.addon_id
  _debug_       = args.debug
  _test_        = args.test
  _use_caffe_   = args.use_caffe
  _cv_model_    = args.cv_model
  _cv_config_   = args.cv_config
  _threshold_   = args.threshold

  if _log_file_:
    logging.basicConfig(filename=_log_file_, format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', filemode='w', level=logging.DEBUG)

  log('===== Starting {} ====='.format(os.path.basename(__file__)))

  log('Output debug info: {}'.format('active' if _debug_ else 'not active'), level='DEBUG')
  log('Log file:          {}'.format(_log_file_ if _log_file_ else 'stdout/syslog'), level='DEBUG')
  #log('Config file:       {}'.format(_config_file_), level='DEBUG')
  log('Addon ID:          {}'.format(_addon_id_), level='DEBUG')

  if not read_config():
    sys.exit(1)

  log('Object Detection:  {}'.format('active' if _attach_analyze_ else 'not active'), level='DEBUG')

  if _attach_analyze_:
    if (_cv_model_ and not os.path.isfile(_cv_model_)) or not _cv_config_:
      _cv_model_ = None
      _cv_config_ = None

    if (_cv_config_ and not os.path.isfile(_cv_config_)) or not _cv_model_:
      _cv_config_ = None
      _cv_model_ = None

    if _use_caffe_:
      #_modelFile_  = _cv_model_ or './models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
      #_configFile_ = _cv_config_ or './models/deploy.prototxt'
      _modelFile_  = _cv_model_ or './models/MobileNetSSD_deploy.caffemodel'
      _configFile_ = _cv_config_ or './models/MobileNetSSD_deploy.prototxt'
      #_net_ = cv2.dnn.readNetFromCaffe(_configFile_, _modelFile_)
    else:
      #_modelFile_  = _cv_model_ or './models/opencv_face_detector_uint8.pb'
      #_configFile_ = _cv_config_ or './models/opencv_face_detector.pbtxt'
      _modelFile_   = _cv_model_ or './models/frozen_inference_graph.pb'
      _configFile_  = _cv_config_ or './models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
      #_net_ = cv2.dnn.readNetFromTensorflow(_modelFile_, _configFile_)

    log('- Method:          {}'.format('Caffe' if _use_caffe_ else 'TensorFlow'), level='DEBUG')
    log('- Threshold:       {}'.format(_threshold_), level='DEBUG')
    log('- Trained Weights: {}'.format(_modelFile_), level='DEBUG')
    log('- Network config:  {}'.format(_configFile_), level='DEBUG')

    log('Loading OpenCV {} model and config for object detection ...'.format('Caffe' if _use_caffe_ else 'TensorFlow'), level='DEBUG')
    if _use_caffe_:
      _net_ = cv2.dnn.readNetFromCaffe(_configFile_, _modelFile_)
    else:
      _net_ = cv2.dnn.readNetFromTensorflow(_modelFile_, _configFile_)
    log('Loading completed.', level='DEBUG')

  if _test_: # Simulate event and send test message to _smtp_user_
    log('Simulating event ...')
    alert(_device_id_[0])
    sys.exit(0)

  # Start the smtp server on port _event_port_
  if not port_is_used(_local_port_):
    log('Listening for event messages on port: {} ...'.format(_local_port_))
    smtp_server = MySMTPServer(('0.0.0.0', _local_port_), None)
  else:
    log('Port {} is already in use.'.format(_local_port_), level='ERROR')
    sys.exit(1)

  try:
    asyncore.loop()

  except (KeyboardInterrupt, SystemExit):
    log('Abort requested by user or system.', level='DEBUG')
    sys.exit(1)

  except Exception as e:
    log('Abort due to exception: \'{}\''.format(e), level='ERROR')
    sys.exit(1)

  finally:
    smtp_server.close()
