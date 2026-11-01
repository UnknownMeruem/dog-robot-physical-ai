from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent
p=Path(__file__).parent
root=Element('mujoco', model='soft_companion_quadruped_v02')
SubElement(root,'compiler', angle='degree', autolimits='true')
SubElement(root,'option', timestep='0.005', gravity='0 0 -9.81', integrator='implicitfast')
defs=SubElement(root,'default')
SubElement(defs,'joint', damping='0.15', armature='0.002')
SubElement(defs,'geom', friction='1.2 0.01 0.001', condim='3')
world=SubElement(root,'worldbody')
SubElement(world,'light', pos='0 -0.8 1.5', dir='0 0 -1')
SubElement(world,'geom',name='floor',type='plane',size='2 2 0.1',rgba='0.86 0.88 0.9 1')
torso=SubElement(world,'body',name='torso',pos='0 0 0.245')
SubElement(torso,'freejoint',name='root')
SubElement(torso,'geom',name='torso_geom',type='box',size='0.155 0.085 0.047',mass='0.75',rgba='0.69 0.46 0.27 1')
# Nose faces negative X. Legs: two hinges per leg, simplified sagittal plane.
for front,x in [('front',-0.112),('hind',0.112)]:
 for side,y in [('left',0.075),('right',-0.075)]:
  tag=f'{front}_{side}'
  hip=SubElement(torso,'body',name=tag+'_upper',pos=f'{x} {y} -0.022')
  SubElement(hip,'joint',name=tag+'_hip',type='hinge',axis='0 1 0',range='-48 48')
  SubElement(hip,'geom',type='capsule',fromto='0 0 0 0 0 -0.087',size='0.019',mass='0.065',rgba='0.75 0.52 0.34 1')
  shin=SubElement(hip,'body',name=tag+'_lower',pos='0 0 -0.087')
  SubElement(shin,'joint',name=tag+'_knee',type='hinge',axis='0 1 0',range='-85 15')
  SubElement(shin,'geom',type='capsule',fromto='0 0 0 0 0 -0.087',size='0.016',mass='0.055',rgba='0.78 0.57 0.38 1')
  SubElement(shin,'geom',name=tag+'_foot',type='sphere',pos='-0.011 0 -0.087',size='0.023',mass='0.018',rgba='0.23 0.19 0.17 1')
neck=SubElement(torso,'body',name='neck_yaw_link',pos='-0.145 0 0.039')
SubElement(
    neck, 'inertial',
    pos='0 0 0',
    mass='0.02',
    diaginertia='0.00002 0.00002 0.00002'
)
SubElement(neck,'joint',name='neck_yaw',type='hinge',axis='0 0 1',range='-35 35')
head=SubElement(neck,'body',name='head',pos='-0.061 0 0.027')
SubElement(head,'joint',name='neck_pitch',type='hinge',axis='0 1 0',range='-22 22')
SubElement(head,'geom',type='ellipsoid',size='0.065 0.065 0.061',mass='0.12',rgba='0.82 0.62 0.42 1')
SubElement(head,'geom',type='sphere',pos='-0.060 0 -0.017',size='0.025',mass='0.012',rgba='0.94 0.8 0.6 1')
for y in [-0.05,0.05]:
 SubElement(head,'geom',type='ellipsoid',pos=f'0 {y} 0.055',size='0.02 0.018 0.044',mass='0.01',rgba='0.56 0.36 0.22 1')
tail=SubElement(torso,'body',name='tail',pos='0.15 0 0.036')
SubElement(tail,'joint',name='tail_yaw',type='hinge',axis='0 0 1',range='-40 40')
SubElement(tail,'geom',type='capsule',fromto='0 0 0 0.09 0 0.043',size='0.017',mass='0.035',rgba='0.79 0.56 0.34 1')
# Breathing visual geometry, non-colliding slider directly above torso.
breath=SubElement(torso,'body',name='breathing_chest',pos='-0.050 0 0.048')
SubElement(breath,'joint',name='chest_expand',type='slide',axis='0 0 1',range='0 0.007',damping='1')
SubElement(breath,'geom',type='ellipsoid',size='0.08 0.07 0.01',mass='0.014',contype='0',conaffinity='0',rgba='0.93 0.76 0.55 0.8')
act=SubElement(root,'actuator')
for front in ['front','hind']:
 for side in ['left','right']:
  for joint,kp in [('hip','15'),('knee','12')]:
   name=f'{front}_{side}_{joint}'
   SubElement(act,'position',name=name+'_drive',joint=name,kp=kp,kv='1.2',forcelimited='true',forcerange='-3 3',inheritrange='1')
for joint,kp,lim in [('neck_yaw','4','-0.6 0.6'),('neck_pitch','4','-0.38 0.38'),('tail_yaw','2','-0.65 0.65'),('chest_expand','80','0 0.007')]:
 SubElement(act,'position',name=joint+'_drive',joint=joint,kp=kp,kv='0.5' if joint!='chest_expand' else '2',ctrllimited='true',ctrlrange=lim)
indent(root)
ElementTree(root).write(p/'dog_quadruped.xml',encoding='unicode',xml_declaration=True)
print('Generated MJCF')
