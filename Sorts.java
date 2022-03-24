public class Sorts{
    
    private static int[] aux; // aux array for mergesort
    
    public static void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    
    public static void print(int[] a){
        for(int i = 0; i < a.length; i++)
            System.out.print(a[i] + " ");
        System.out.println();
        System.out.println();
    }
    
    public static void selectionSort(int[] nums){
        for(int i = 0; i < nums.length; i++)
            for (int j = i + 1; j < nums.length; j++)
                if(nums[j] < nums[i])
                    swap(nums, i, j);
    }
    
    public static void insertionSort(int[] nums){
        for(int i = 1; i < nums.length; i++)
            for(int j = i; j > 0 && nums[j] < nums[j-1]; j--)
                swap(nums, j, j-1);
    }
    
    public static void shellSort(int[] nums){
        int h = 1;
        while(h < nums.length / 3) h = 3 * h + 1;
        while(h >= 1){
            for(int i = h; i < nums.length; i++)
                for(int j = i; j >= h && nums[j] < nums[j-h]; j = j - h)
                    swap(nums, j, j-h);
            h /= 3;
        }
    }
    
    private static void merge(int[] a, int lo, int mid, int hi){
        int i = lo;
        int j = mid + 1;
        for(int k = 0; k < hi + 1; k++)
            aux[k] = a[k];
        for(int k = lo; k < hi + 1; k++){
            if(i > mid)              a[k] = aux[j++];
            else if(j > hi)          a[k] = aux[i++];
            else if(aux[i] < aux[j]) a[k] = aux[i++];
            else                     a[k] = aux[j++];
        }
    }
    
    public static void mergesort(int[] nums){
        aux = new int[nums.length]; // aux array for mergesort
        mergesort(nums, 0, nums.length - 1);
    }
    
    private static void mergesort(int[] nums, int lo, int hi){
        if(hi <= lo) return;
        int mid = lo + (hi - lo) / 2;
        mergesort(nums, lo, mid);
        mergesort(nums, mid + 1, hi);
        merge(nums, lo, mid, hi);
    }
    
    private static int partition(int[] nums, int lo, int hi){
        int j = lo;
        int pivot = nums[hi];
        for(int i = lo; i < hi; i++){
            if(nums[j] < pivot){
                swap(nums, i, j);
                j++;
            }
        }
        swap(nums,j,hi);
        return j;
    }

    
    private static void quicksort(int[] nums, int lo, int hi){
        int j = partition(nums, lo, hi);
        quicksort(nums, lo, j);
        quicksort(nums, j+1, hi);
    }

        
    public static void quicksort(int[] nums){
        quicksort(nums, 0, nums.length-1);
    }
    
    public static void mergesortBU(int[] nums){
        aux = new int[nums.length];
        for(int sz = 1; sz < nums.length; sz *= 2){
            for(int lo = 0; lo < nums.length - sz; lo += 2 * sz)
                merge(nums, lo, lo + sz - 1, Math.min(lo + 2 * sz - 1, nums.length - 1));
        }
    }
    
    
    public static void main(String []args){
        
        int[] test1 = new int[] { 5, 4, 1, 7, 3, 6, 2, 8, 5, 10 };
        int[] test2 = new int[] { 5, 4, 1, 7, 3, 6, 2, 8, 5, 10 };
        int[] test3 = new int[] { 5, 4, 1, 7, 3, 6, 2, 8, 5, 10 };
        int[] test4 = new int[] { 5, 4, 1, 7, 3, 6, 2, 8, 5, 10 };
        int[] test5 = new int[] { 5, 4, 1, 7, 3, 6, 2, 8, 5, 10 };
        int[] test6 = new int[] { 5, 4, 1, 7, 3, 6, 2, 8, 5, 10 };
        
        System.out.println("Original Array");
        print(test1);
        
        System.out.println("Selection sort");
        selectionSort(test1);
        print(test1);
        
        System.out.println("Insertion sort");
        insertionSort(test2);
        print(test2);
        
        System.out.println("Shell sort");
        shellSort(test3);
        print(test3);
        
        System.out.println("Mergesort");
        mergesort(test4);
        print(test4);
        
        System.out.println("Quicksort");
        mergesort(test5);
        print(test5);        
        
        System.out.println("BU Mergesort");
        mergesortBU(test6);
        print(test6);
    }
}