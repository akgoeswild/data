import java.math.BigInteger;
import java.util.Scanner;

public class Assignment2 {
	public static int inv(int a, int m) {
		return BigInteger.valueOf(a).modInverse(BigInteger.valueOf(m)).intValue();
	}
	public static int crt(int [] num, int [] rem, int k) {
		
		int prod = 1;
		for(int i=0; i<k; i++) {
			prod*=num[i];
		}
		int result = 0;
		for(int i=0; i<k; i++) {
			int pp = prod/num[i];
			result+=pp*inv(pp,num[i])*rem[i];
		}
		return result%prod;
	}
	public static void main(String [] args) {
		Scanner sc = new Scanner(System.in);
		int k = sc.nextInt();
		int num [] = new int[k];
		int rem [] = new int[k];
		for(int i=0; i<k; i++) {
			num[i] = sc.nextInt();
			rem[i] = sc.nextInt();
		}
		System.out.println("X: "+crt(num, rem, k));
		sc.close();
	}
}
