#!/usr/bin/env python3
"""arm_ik_verify
6축 매니퓰레이터(yaw-pitch-pitch-roll-pitch-roll, 구형 손목) DH 역기구학 검증 노드.

robot_description / joint_states 는 외부에서 발행되는 것을 구독한다.
  1) robot_description(URDF) FK  -> x, y, z, alpha, beta, gamma
  2) 같은 관절각을 DH 테이블에 대입한 FK -> (1)과 비교
  3) (1)의 pose 로 DH 기반 IK -> arm_joint1~6 과 비교
  4) IK 결과를 다시 FK 로 돌려 재현 오차 출력
"""
import math
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

D2R = math.pi / 180.0
R2D = 180.0 / math.pi

# d[mm], theta_offset[deg], a[mm], alpha[deg]   (L3 = hypot(a3, d4) 로 자동 유도)
DH_DEFAULT = [171.0,    0.0000,   0.000, -90.0,
              0.0,    -86.3062, 416.878,   0.0,
              0.0,    -93.6938,  27.172, -90.0,
              422.727,  0.0000,   0.000,  90.0,
              0.0,      0.0000,   0.000, -90.0,
              0.0,     90.0227,   0.000,   0.0]

JOINTS = ['arm_joint1', 'arm_joint2', 'arm_joint3',
          'arm_joint4', 'arm_joint5', 'arm_joint6']


# ============================ 수학 유틸 ============================
def Rx(t): c, s = math.cos(t), math.sin(t); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(t): c, s = math.cos(t), math.sin(t); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(t): c, s = math.cos(t), math.sin(t); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def wrap(t): return (t + math.pi) % (2 * math.pi) - math.pi


def abg_to_R(al, be, ga):
    """alpha(z) beta(y) gamma(x) -> R = Rz(al)Ry(be)Rx(ga)"""
    return Rz(al) @ Ry(be) @ Rx(ga)


def R_to_abg(R):
    """R -> (alpha, beta, gamma)"""
    be = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if abs(math.cos(be)) < 1e-9:
        return np.array([0.0, be, math.atan2(R[0, 1], R[1, 1])])
    return np.array([math.atan2(R[1, 0], R[0, 0]), be,
                     math.atan2(R[2, 1], R[2, 2])])


def dh_mat(th, d, a, al):
    """표준 DH :  Rz(th) Tz(d) Tx(a) Rx(al)"""
    ct, st, ca, sa = math.cos(th), math.sin(th), math.cos(al), math.sin(al)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                     [st,  ct*ca, -ct*sa, a*st],
                     [0.,  sa,     ca,    d   ],
                     [0.,  0.,     0.,    1.  ]])


# ========================= DH 기반 FK / IK =========================
class DhArm:
    """2R 상수와 theta5 부호를 DH 테이블에서 자동 유도"""

    def __init__(self, table, tool_len, base_xyz=(0.0, 0.0, 0.0)):
        self.DH = [tuple(float(v) for v in row) for row in table]  # (d, off, a, al)
        self.L = float(tool_len)
        self.base = np.asarray(base_xyz, float)
        self.L1 = self.DH[0][0]
        self.L2 = self.DH[1][2]
        self.a3 = self.DH[2][2]
        self.d4 = self.DH[3][0]
        self.L3 = math.hypot(self.a3, self.d4)      # 381.695
        self.DELTA = math.atan2(self.d4, self.a3)   # 85.9178 deg
        self.OFF2 = self.DH[1][1]
        self.OFF3 = self.DH[2][1]
        self.OFF6 = self.DH[5][1]
        # Rx(al4) Rz(t5) Rx(al5) 가 Ry(+t5) 인지 Ry(-t5) 인지
        self.S5 = -1.0 if math.sin(self.DH[3][3]) > 0 else 1.0

    def T0i(self, q, i):
        T = np.eye(4)
        T[:3, 3] = self.base
        for k in range(i):
            d, off, a, al = self.DH[k]
            T = T @ dh_mat(q[k] + off, d, a, al)
        return T

    def fk(self, q):
        T = self.T0i(q, 6)
        A = np.eye(4)
        A[2, 3] = self.L
        return T @ A

    def ik(self, p, R, q_prev=None):
        pw = np.asarray(p, float) - self.L * R[:, 2] - self.base   # 손목중심
        rho = math.hypot(pw[0], pw[1])
        s = pw[2] - self.L1
        sols = []
        for front in (1, -1):
            if rho < 5.0 and q_prev is not None:                   # 어깨 특이점
                t1 = q_prev[0]
                rh = pw[0]*math.cos(t1) + pw[1]*math.sin(t1)
            else:
                t1 = wrap(math.atan2(pw[1], pw[0])
                          + (0.0 if front > 0 else math.pi))
                rh = front * rho
            c = (rh*rh + s*s - self.L2**2 - self.L3**2) / (2*self.L2*self.L3)
            if abs(c) > 1 + 1e-5:                                  # 톨러런스 1e-5
                continue
            c = max(-1.0, min(1.0, c))
            for el in (1, -1):
                sx = el * math.sqrt(max(0.0, 1.0 - c*c))
                xi = math.atan2(sx, c)                             # 엘보 turn 각
                # alpha1 = -90 -> frame1 의 y1 = -z  =>  평면각 = atan2(-s, rh)
                TH2 = math.atan2(-s, rh) \
                      - math.atan2(self.L3*sx, self.L2 + self.L3*c)
                t2 = wrap(TH2 - self.OFF2)
                t3 = wrap(xi - self.DELTA - self.OFF3)
                R03 = self.T0i([t1, t2, t3, 0, 0, 0], 3)[:3, :3]
                M = R03.T @ R @ Rz(-self.OFF6)
                sp = math.hypot(M[0, 2], M[1, 2])
                if sp < 1e-9:                                      # 손목 특이점
                    t5 = 0.0 if M[2, 2] > 0 else math.pi
                    t4 = 0.0 if q_prev is None else q_prev[3]
                    t6 = wrap(math.atan2(M[1, 0], M[0, 0]) - t4) if M[2, 2] > 0 \
                        else wrap(math.atan2(M[1, 0], M[1, 1]) + t4)
                    sols.append(np.array([t1, t2, t3, t4, self.S5*t5, t6]))
                    continue
                for k in (1, -1):
                    sols.append(np.array([
                        t1, t2, t3,
                        wrap(math.atan2(k*M[1, 2], k*M[0, 2])),
                        wrap(self.S5 * math.atan2(k*sp, M[2, 2])),  # 부호 반전
                        wrap(math.atan2(k*M[2, 1], -k*M[2, 0]))]))
        return sols

    def info(self):
        return (f'L1={self.L1:.3f} L2={self.L2:.3f} '
                f'L3=hypot({self.a3:.3f},{self.d4:.3f})={self.L3:.3f} '
                f'L={self.L:.3f} [mm]\n'
                f'  delta={self.DELTA*R2D:.4f} off2={self.OFF2*R2D:+.4f} '
                f'off3={self.OFF3*R2D:+.4f} off6={self.OFF6*R2D:+.4f} [deg], '
                f'theta5 부호={self.S5:+.0f}\n'
                f'  reach=[{abs(self.L2-self.L3):.3f}, {self.L2+self.L3:.3f}] mm, '
                f'완전신장 theta3={(math.pi-self.DELTA-self.OFF3)*R2D:.4f} deg')


# ===================== robot_description(URDF) FK =====================
class UrdfChain:
    """URDF 에서 x,y,z,alpha,beta,gamma 와 관절 한계를 읽는다 (길이 단위 mm 로 변환)"""

    def __init__(self, urdf_xml, base='mani_base_link', tip='tcp_link'):
        self.J = self._parse(urdf_xml)
        self.chain = self._chain(base, tip)

    @staticmethod
    def _parse(xml_text):
        root = ET.fromstring(xml_text)
        out = {}
        for j in root.findall('joint'):
            o, ax, lim = j.find('origin'), j.find('axis'), j.find('limit')
            xyz = [0., 0., 0.]
            rpy = [0., 0., 0.]
            if o is not None:
                if o.get('xyz'):
                    xyz = [float(v) for v in o.get('xyz').split()]
                if o.get('rpy'):
                    rpy = [float(v) for v in o.get('rpy').split()]
            axis = [float(v) for v in ax.get('xyz').split()] \
                if ax is not None else [0., 0., 1.]
            lo = float(lim.get('lower')) \
                if lim is not None and lim.get('lower') else -math.pi
            up = float(lim.get('upper')) \
                if lim is not None and lim.get('upper') else math.pi
            out[j.get('name')] = dict(
                type=j.get('type'), lower=lo, upper=up,
                parent=j.find('parent').get('link'),
                child=j.find('child').get('link'),
                xyz=np.array(xyz) * 1000.0,            # m -> mm
                rpy=np.array(rpy), axis=np.array(axis))
        return out

    def _chain(self, base, tip):
        names, cur = [], tip
        for _ in range(64):
            f = [n for n, v in self.J.items() if v['child'] == cur]
            if not f:
                break
            names.append(f[0])
            cur = self.J[f[0]]['parent']
            if cur == base:
                break
        if cur != base:
            raise RuntimeError(f'{base} -> {tip} 경로 없음')
        return list(reversed(names))

    def _T(self, name, q=0.0):
        v = self.J[name]
        R = Rz(v['rpy'][2]) @ Ry(v['rpy'][1]) @ Rx(v['rpy'][0])
        if v['type'] in ('revolute', 'continuous'):
            a = v['axis'] / np.linalg.norm(v['axis'])
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            R = R @ (np.eye(3) + math.sin(q)*K + (1 - math.cos(q))*(K @ K))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = v['xyz']
        return T

    def fk(self, q):
        T = np.eye(4)
        k = 0
        for n in self.chain:
            if self.J[n]['type'] in ('revolute', 'continuous'):
                T = T @ self._T(n, q[k])
                k += 1
            else:
                T = T @ self._T(n)
        return T

    def limits(self):
        return np.array([[self.J[n]['lower'], self.J[n]['upper']] for n in JOINTS])


# ============================== 노드 ==============================
class ArmIkVerify(Node):
    def __init__(self):
        super().__init__('arm_ik_verify')
        self.declare_parameter('base_link', 'mani_base_link')
        self.declare_parameter('tip_link', 'tcp_link')
        self.declare_parameter('dh', DH_DEFAULT)
        self.declare_parameter('tool_length', 133.799)
        self.declare_parameter('base_xyz', [27.0, 27.0, 0.0])
        self.declare_parameter('print_period', 1.0)

        flat = list(self.get_parameter('dh').value)
        if len(flat) != 24:
            raise ValueError('dh 파라미터는 6x4 = 24개여야 합니다')
        table = [(flat[4*i], flat[4*i+1]*D2R, flat[4*i+2], flat[4*i+3]*D2R)
                 for i in range(6)]
        self.arm = DhArm(table,
                         self.get_parameter('tool_length').value,
                         self.get_parameter('base_xyz').value)
        self.get_logger().info('[DH] ' + self.arm.info())

        self.period = float(self.get_parameter('print_period').value)
        self.urdf = None
        self.q_prev = None
        self.t_last = 0.0

        qos = QoSProfile(depth=1,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(String, '/robot_description', self._on_urdf, qos)
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)
        self.pub = self.create_publisher(JointState, '~/ik_result', 10)
        self.get_logger().info('/robot_description 과 /joint_states 대기 중...')

    # ---------- 구독 ----------
    def _on_urdf(self, msg):
        if self.urdf is not None:
            return
        try:
            self.urdf = UrdfChain(msg.data,
                                  self.get_parameter('base_link').value,
                                  self.get_parameter('tip_link').value)
        except Exception as e:
            self.get_logger().error(f'URDF 파싱 실패: {e}')
            return
        self.get_logger().info('[URDF] chain : ' + ' -> '.join(self.urdf.chain))

    def _on_js(self, msg):
        if self.urdf is None:
            return
        try:
            q = np.array([msg.position[msg.name.index(n)] for n in JOINTS])
        except (ValueError, IndexError):
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        if t - self.t_last < self.period:
            return
        self.t_last = t
        self._report(q)

    # ---------- 본체 ----------
    def _report(self, q):
        arm = self.arm
        lim = self.urdf.limits()

        Tu = self.urdf.fk(q)
        pu, Ru = Tu[:3, 3], Tu[:3, :3]
        au = R_to_abg(Ru)

        Td = arm.fk(q)
        pd, Rd = Td[:3, 3], Td[:3, :3]
        ad = R_to_abg(Rd)

        allsol = arm.ik(pu, Ru, self.q_prev)
        inlim = [x for x in allsol
                 if np.all((x >= lim[:, 0] - 1e-9) & (x <= lim[:, 1] + 1e-9))]
        pool = inlim if inlim else allsol
        if not pool:
            self.get_logger().warn('IK 해 없음 (도달 불가)')
            return
        ref = self.q_prev if self.q_prev is not None else q
        w = np.array([3., 2., 2., 1., 1., 1.])
        qik = min(pool, key=lambda x: float(
            w @ np.abs(np.array([wrap(a - b) for a, b in zip(x, ref)]))))
        self.q_prev = qik

        Tr = arm.fk(qik)
        dp = float(np.linalg.norm(Tr[:3, 3] - pu))
        dR = float(np.linalg.norm(Tr[:3, :3].T @ Ru - np.eye(3)))

        L = ['', '=' * 72,
             f'[pose]  단위 mm / deg      IK 해 {len(allsol)}개 '
             f'(한계통과 {len(inlim)}개)',
             f'  {"":10s}{"x":>10s}{"y":>10s}{"z":>10s}'
             f'{"alpha":>10s}{"beta":>10s}{"gamma":>10s}',
             f'  {"URDF":10s}' + ''.join(f'{v:10.3f}' for v in pu)
             + ''.join(f'{v*R2D:10.4f}' for v in au),
             f'  {"DH":10s}' + ''.join(f'{v:10.3f}' for v in pd)
             + ''.join(f'{v*R2D:10.4f}' for v in ad),
             f'  {"diff":10s}' + ''.join(f'{a-b:10.3f}' for a, b in zip(pd, pu))
             + ''.join(f'{wrap(a-b)*R2D:10.4f}' for a, b in zip(ad, au)),
             '-' * 72,
             f'[joint] {"":12s}{"URDF/js":>12s}{"DH-IK":>12s}'
             f'{"diff[deg]":>12s}{"limit[deg]":>18s}']
        for i, n in enumerate(JOINTS):
            d = wrap(qik[i] - q[i]) * R2D
            flag = '' if abs(d) < 0.05 else '  <-- 불일치'
            L.append(f'  {n:18s}{q[i]*R2D:12.4f}{qik[i]*R2D:12.4f}{d:12.4f}'
                     f'   [{lim[i,0]*R2D:7.2f},{lim[i,1]*R2D:7.2f}]{flag}')
        L += ['-' * 72,
              f'[check] DH-IK -> DH-FK 재현 : 위치 {dp*1000:.3f} um, 자세 {dR:.2e}',
              '=' * 72]
        self.get_logger().info('\n'.join(L))

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(JOINTS)
        out.position = [float(v) for v in qik]
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ArmIkVerify()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
