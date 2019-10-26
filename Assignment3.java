import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Scanner;

public class Assignment3 {
	public static String encrypt(String input) {
		try {
			MessageDigest md = MessageDigest.getInstance("SHA-1");
			byte [] messageDigest = md.digest(input.getBytes());
			BigInteger no = new BigInteger(1,messageDigest);
			String hashtext = no.toString(16);
			if(hashtext.length()<32) {
				hashtext = "0"+hashtext;
			}
			
			return hashtext;
		} catch (NoSuchAlgorithmException e) {
			// TODO Auto-generated catch block
			throw new RuntimeException(e);
		}
		
	}
	public static void main(String [] args) {
		Scanner sc = new Scanner(System.in);
		String s = sc.nextLine();
		System.out.println(encrypt(s));
		sc.close();
	}
}
