
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
import ros2_numpy as rnp
import matplotlib.pyplot as plt


class ImgTester(Node):
    def __init__(self):
        super().__init__('img_test') 
        self.img_pub = self.create_publisher(
            Image,
            "/test_img",
            10
        )

        self.img_sub = self.create_subscription(
            Image,
            "/test_img",
            self.img_callback,    
            10
        )

        # send an image to get the chain started
        self._send_rand_img()

    def img_callback(self, msg: Image):
        # show sent
        plt.imshow(self.latest_sent)
        plt.title("sent")
        plt.show()

        print(msg.header.stamp)

        # show received
        plt.imshow(rnp.numpify(msg))
        plt.title("received")
        plt.show()

        self._send_rand_img()

    def _send_rand_img(self):
        arr = np.random.randint(0, 255, (300,486,3)).astype(np.uint8)
        self.latest_sent = arr

        msg = rnp.msgify(Image, arr, encoding="rgb8")
        msg.header.stamp = self.get_clock().now().to_msg()
        
        self.img_pub.publish(msg)


if __name__ == "__main__":
    rclpy.init(args = None)
    it = ImgTester()
    rclpy.spin(it)