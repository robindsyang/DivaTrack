import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

class MVNX:
    """
    The abstract parser object to run through the XML tree structure of the MVNX file format
    and extract the relevant information into dictionaries and numpy arrays. Super simple, needs
    refactoring at the moment.
    Can also be used as a command line tool.
    """
    def __init__(self, path, orientation=None, position=None, velocity=None, \
                       acceleration=None, angularVelocity=None, angularAcceleration=None, \
                       footContacts=None, sensorFreeAcceleration=None, sensorMagneticField=None, \
                       sensorOrientation=None, jointAngle=None, jointAngleXZY=None, jointAngleErgo=None, \
                       centerOfMass=None, mapping=None, sensors=None, segments=None, joints=None, \
                       root=None, mvn=None, comment=None, subject=None, version=None, build=None, label=None, \
                       frameRate=None, segmentCount=None, recordingDate=None, configuration=None, userScenario=None, \
                       securityCode=None, modality=None, time=None, index=None, timecode=None, ms=None):
        if orientation is None:                       
            self.orientation = []
        if position is None:
            self.position = []
        if velocity is None:
            self.velocity = []
        if acceleration is None:    
            self.acceleration = []
        if angularVelocity is None:
            self.angularVelocity = []
        if angularAcceleration is None:
            self.angularAcceleration = []
        if footContacts is None:
            self.footContacts = []
        if sensorFreeAcceleration is None:
            self.sensorFreeAcceleration = []
        if sensorMagneticField is None:
            self.sensorMagneticField = []
        if sensorOrientation is None:
            self.sensorOrientation = []
        if jointAngle is None:
            self.jointAngle = []
        if jointAngleXZY is None:
            self.jointAngleXZY = []
        if jointAngleErgo is None:
            self.jointAngleErgo = []
        if centerOfMass is None:
            self.centerOfMass = []
        if sensors is None:
            self.sensors = []
        if segments is None:
            self.segments = {}
        if joints is None:
            self.joints = {} 
        if mapping is None:
            self.mapping = {"orientation": 0,
                            "position": 1,                       
                            "footContacts": 2, 
                            "sensorFreeAcceleration": 3}
        if time is None:
            self.time = []
        else:
            self.time = time
        if index is None:
            self.index = []
        else:
            self.index = index 
        if timecode is None:
            self.timecode = []
        else:
            self.timecode = timecode
        if ms is None:
            self.ms = []
        else:
            self.ms = ms
        self.mvn = mvn
        self.comment = comment
        self.subject = subject
        self.version = version
        self.build = build
        self.label = label
        self.frameRate = frameRate
        self.segmentCount = segmentCount
        self.recordingDate = recordingDate
        self.configuration = configuration
        self.userScenario = userScenario
        self.securityCode = securityCode
        self.modality = modality
        if path is None:
            print('Please supply a path')
        self.path = path
        if root is None:
            self.parse_mvnx(self.path)
            self.footContacts = self.parse_modality('footContacts')
            self.sensorFreeAcceleration = self.parse_modality('sensorFreeAcceleration')
        else:
            self.root = root

    def __repr__(self):
        return f'<MVNX ({self.path})>'
             
      
    def parse_mvnx(self, path):
        """
        Take a path to an MVNX file and parse it
        Args:
            path ([string]): [the path to the data file]
        """
        tree = ET.parse(path)
        self.root = tree.getroot()
        self.mvn = self.root[0]
        self.version = self.root[0].attrib['version']
        self.build = self.root[0].attrib['build']
        self.comment = self.root[1].text
        self.label = self.root[2].attrib['label']
        self.frameRate = self.root[2].attrib['frameRate']
        self.segmentCount = self.root[2].attrib['segmentCount']
        self.recordingDate = self.root[2].attrib['recDate']
        self.configuration = self.root[2].attrib['configuration']
        self.userScenario = self.root[2].attrib['userScenario']
        self.securityCode = self.root[3].attrib['code']
        return self.root
    
    def parse_modality(self, modality):

        """[With a given XML Tree, parse out the salient modalities within each frame]
        Args:
            modality ([string]): [the name of the modality]
        """

        holding_list = []
        frames = self.root[2][4]
        for frame in tqdm(frames[3:]):
            for child in frame[self.mapping[modality]:self.mapping[modality]+1]:
                holding_list.append(child.text.split(' '))           
        holding_list = np.array(holding_list)
        return holding_list.astype(float)

    def parse_time(self):
        frames = self.root[2][4][3:]
        for frame in tqdm(frames):
            self.time.append(frame.attrib['time'])
        return self.time
    
    def parse_index(self):
        frames = self.root[2][4][3:]
        for frame in tqdm(frames):
            self.index.append(frame.attrib['index'])
        return self.index
    
    def parse_timecode(self):
        frames = self.root[2][4][3:]
        for frame in tqdm(frames):
            self.timecode.append(frame.attrib['tc'])
        return self.timecode
    
    def parse_ms(self):
        frames = self.root[2][4][3:]
        for frame in tqdm(frames):
            self.ms.append(frame.attrib['ms'])
        return self.ms

    def parse_modalities(self, *args):
        for arg in tqdm(args):
            print(self.parse_modality(arg))
            return self.parse_modality(arg)

    def parse_sensors(self):
        for sensor in tqdm(self.root[2][2]):
            self.sensors.append(sensor.attrib['label'])
        return self.sensors
    
    def parse_segments(self):
        for segment in tqdm(self.root[2][1]):
            self.segments[segment.attrib['id']] = segment.attrib['label']
        return self.segments
    
    def parse_joints(self):
        for joint in tqdm(self.root[2][3]):
            self.joints[joint.attrib['label']] = [joint[0].text, joint[1].text]
        return self.joints

    def parse_all(self):
        for key in tqdm(self.mapping.keys()):
            setattr(self, key, self.parse_modality(key))
        self.parse_time()
        self.parse_joints()
        self.parse_segments()
        self.parse_sensors()
        self.parse_timecode()
        self.parse_ms()

if __name__ == '__main__':
    mvnx = MVNX('/home/robindsyang/Projects/Crayfish_python_V2/data_raw/meta_hardcases/S3_016.mvnx')
    mvnx.parse_all()
    print(mvnx.footContacts.shape)
    print(mvnx.sensorFreeAcceleration.shape)