import java.math.BigInteger;
import java.util.Scanner;

public class Assignment1 {
	public static void main(String [] args) {
		Scanner sc = new Scanner(System.in);
		int p = sc.nextInt();
		int q = sc.nextInt();
		int pubKey = 6;
		BigInteger big_p = new BigInteger(""+p);
		BigInteger big_q = new BigInteger(""+q);
		BigInteger big_n = big_p.multiply(big_q);
		BigInteger big_p_1 = big_p.subtract(new BigInteger("1"));
		BigInteger big_q_1 = big_q.subtract(new BigInteger("1"));
		BigInteger big_p_1_q_1 = big_p_1.multiply(big_q_1);
		
		//Generating public key
		while(true) {
			BigInteger big_gcd = big_p_1_q_1.gcd(new BigInteger(""+pubKey));
			if(big_gcd.equals(BigInteger.ONE)) {
				break;
			}
			pubKey++;
		}
		BigInteger big_pubKey = new BigInteger(""+pubKey);
		BigInteger big_prvKey = big_pubKey.modInverse(big_p_1_q_1);
		System.out.println("Public Key: "+big_pubKey+" "+big_n);
		System.out.println("Private Key: "+big_prvKey+" "+big_n);
		
		int msg = sc.nextInt();
		BigInteger big_msg = new BigInteger(""+msg);
		
		BigInteger big_cipherVal = big_msg.modPow(big_pubKey, big_n);
		BigInteger big_plainVal = big_cipherVal.modPow(big_prvKey, big_n);
		int plainVal = big_plainVal.intValue();
		System.out.println("Cipher value: "+big_cipherVal);
		System.out.println("Plain value: "+ plainVal);
		sc.close();
	}
}
