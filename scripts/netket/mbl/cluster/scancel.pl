use File::Copy;
use Data::Dumper;
use Cwd;
$dir = getcwd;

 

for($val = 2397091; ($val <= 2397102); $val+=1)
{
	system "scancel $val";
}